# ---------------------------------------------------------------------------------------------------
# CLIP-DINOiser: Stratified Training
# ---------------------------------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import transforms

import argparse
import os
import random
import time
import json
import cv2
from datetime import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as T
from hydra import compose, initialize
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist
from mmseg.apis import multi_gpu_test
from tqdm import tqdm

from feature_utils.data_feature.registry import build_feature
from helpers.logger import get_logger
from helpers.trainability import (
    build_optimizer_groups,
    collect_module_grad_summaries,
    configure_trainable_modules,
    normalize_module_paths,
    set_train_mode_for_modules,
)
from models import build_model
from scheduler import MultiStepLR
from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_seg_inference
from feature_utils.data_feature.implementations.quality import LaplacianSharpness, BoundaryGradientAdherence
from feature_utils.data_feature.sampler import StratifiedSampler
from feature_utils.visualizer import plot_metric_distribution, plot_class_distribution_comparison
try:
    from validation_acceleration import (
        build_validation_payload,
        is_cuda_oom_error,
        resolve_proxy_test_cfg,
        sample_dataset_basenames,
        subset_dataset_by_basenames,
    )
except ImportError:
    from clip_dinoiser.validation_acceleration import (
        build_validation_payload,
        is_cuda_oom_error,
        resolve_proxy_test_cfg,
        sample_dataset_basenames,
        subset_dataset_by_basenames,
    )
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from feature_utils.visualizer import plot_metric_distribution, plot_class_distribution_comparison


def get_model_dict(model, parameter_names: list[str] | None = None):
    state_dict = model.state_dict()
    if parameter_names:
        return {name: state_dict[name].cpu() for name in parameter_names if name in state_dict}
    new_check = {}
    new_check['obj_proj.bias'] = state_dict['obj_proj.bias'].cpu()
    new_check['obj_proj.weight'] = state_dict['obj_proj.weight'].cpu()
    new_check['bkg_decoder.bias'] = state_dict['bkg_decoder.bias'].cpu()
    new_check['bkg_decoder.weight'] = state_dict['bkg_decoder.weight'].cpu()
    return new_check


def get_criterion(cfg):
    if cfg.get('loss') == 'CE':
        return torch.nn.BCEWithLogitsLoss(reduction='mean')
    else:
        raise NotImplementedError
def visualize_experiment_results(json_files):
    all_summary = []
    all_per_class = []
    for strategy, path in json_files.items():
        if not os.path.exists(path): continue
        with open(path, 'r') as f:
            data = json.load(f)
            task_data = data.get('coco_stuff', {})
            summary = task_data.get('summary', {})
            summary['Strategy'] = strategy
            all_summary.append(summary)
            for cls, metrics in task_data.get('per_class', {}).items():
                all_per_class.append({'Strategy': strategy, 'Class': cls, 'IoU': metrics['IoU']})
    
    if not all_summary: return
    df_s = pd.DataFrame(all_summary)
    df_c = pd.DataFrame(all_per_class)

    # Plot 2: mIoU Comparison
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_s, x='Strategy', y='mIoU', palette='viridis')
    plt.title("Plot 2: Global mIoU Comparison (%)")
    plt.savefig("plot_2_global_comparison.png")

    # Plot 3: Class Gaps
    pivot_df = df_c.pivot(index='Class', columns='Strategy', values='IoU')
    pivot_df['Gap'] = pivot_df.max(axis=1) - pivot_df.min(axis=1)
    top_gap_classes = pivot_df.sort_values('Gap', ascending=False).head(20).index
    plt.figure(figsize=(12, 8))
    sns.pointplot(data=df_c[df_c['Class'].isin(top_gap_classes)], y='Class', x='IoU', hue='Strategy', join=False, dodge=0.3)
    plt.title("Plot 3: Per-class IoU Gaps (Top 20)")
    plt.tight_layout()
    plt.savefig("plot_3_class_gaps.png")



def do_train(model, train_cfg, loaders, out_path):
    rank, _ = get_dist_info()
    timestamp = time.time()
    date_time = datetime.fromtimestamp(timestamp)
    str_date_time = date_time.strftime("%d%m%Y-%H%M%S")

    # 关键修复：只允许 Rank 0 创建文件夹
    ch_path = os.path.join(out_path, str_date_time)
    if rank == 0:
        os.makedirs(ch_path, exist_ok=True)
    
    # 同步一下，确保其他进程在文件夹创建后再开始训练（可选）
    dist.barrier()
    
    model.to("cuda")
    epochs = train_cfg.get("epochs", 100)
    criterion = get_criterion(train_cfg)
    trainable_modules = normalize_module_paths(train_cfg.get("trainable_modules"))
    active_modules = configure_trainable_modules(model, trainable_modules)
    optimizer_groups, trainability_summary = build_optimizer_groups(
        model,
        corr_lr=float(train_cfg.get('corr_lr')),
        found_lr=float(train_cfg.get('found_lr')),
        backbone_lr=float(train_cfg.get("backbone_lr", float(train_cfg.get('corr_lr')) * 0.2)),
    )
    trainability_summary.trainable_modules = list(active_modules)
    if rank == 0:
        print(
            "[Trainability] modules="
            + ",".join(active_modules)
            + f" trainable_params={len(trainability_summary.trainable_param_names)}",
            flush=True,
        )
    optimizer = torch.optim.AdamW(optimizer_groups, lr=train_cfg.get('corr_lr'))
    scheduler = MultiStepLR(optimizer, train_cfg.get('milestones'), gamma=train_cfg.get("step_lr_gamma"), warmup=0)

    for epoch in range(epochs):
        tbar = tqdm(enumerate(loaders['train'], 0), disable=(get_dist_info()[0] != 0))
        for i, data in tbar:
            set_train_mode_for_modules(model, active_modules)
            inputs = data[0].to("cuda")
            optimizer.zero_grad()
            preds_bkg, pred_corrs, clip_feats = model.forward_pass(inputs)
            pred_corrs[pred_corrs < 0] = 0.

            with torch.no_grad():
                found_pred = model.get_found_preds(inputs, resize=preds_bkg.shape[-2:])
                found_pred = (found_pred > 0.5).float()
                dino_corrs = model.get_dino_corrs(inputs).detach()

            dino_loss = criterion(pred_corrs.float().flatten(-2, -1), (dino_corrs.flatten(-2, -1) > 0).float())
            found_loss = criterion(preds_bkg.float().flatten(-2, -1), found_pred.float().flatten(-2, -1))
            loss = dino_loss + found_loss
            loss.backward()
            if rank == 0 and epoch == 0 and i == 0:
                grad_summaries = collect_module_grad_summaries(model, active_modules)
                grad_parts = [
                    f"{item.module_path}:params_with_grad={item.params_with_grad}/{item.param_count},grad_norm={item.total_grad_norm:.6f}"
                    for item in grad_summaries
                ]
                print("[TrainabilityGrad] " + " | ".join(grad_parts), flush=True)
            optimizer.step()

            if get_dist_info()[0] == 0:
                tbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            scheduler.step()

    # save checkpoint
    if rank == 0:
        model.found_model = None
        model.vit_encoder = None
        torch.save({
            'epoch': epoch, # type:ignore
            'trainable_modules': list(active_modules),
            'trainable_param_names': list(trainability_summary.trainable_param_names),
            'model_state_dict': get_model_dict(model, list(trainability_summary.trainable_param_names)),
        }, os.path.join(ch_path, 'last.pt'))


def _validation_mode(cfg) -> str:
    return str(cfg.evaluate.get("validation_mode", "proxy"))


def _eval_workers_per_gpu(cfg) -> int:
    return int(cfg.evaluate.get("workers_per_gpu", 4))


def build_eval_dataset(key, mode, cfg):
    dataset = build_seg_dataset(cfg.evaluate.get(key))
    if mode != "proxy":
        return dataset

    proxy_subset_size = int(cfg.evaluate.get("proxy_subset_size", 300))
    proxy_seed = int(cfg.evaluate.get("proxy_seed", cfg.seed))
    keep_basenames = sample_dataset_basenames(dataset, seed=proxy_seed, limit=proxy_subset_size)
    subset_dataset_by_basenames(dataset, keep_basenames)
    return dataset


def _run_seg_eval(
    model,
    loader,
    eval_key,
    logger,
    cfg,
    *,
    validation_mode,
):
    model.clip_backbone.decode_head.update_vocab(loader.dataset.CLASSES)
    requested_inference_mode = "slide"
    test_cfg_override = None
    if validation_mode == "proxy":
        requested_inference_mode = str(cfg.evaluate.get("proxy_inference_mode", "whole"))
        test_cfg_override = resolve_proxy_test_cfg(requested_inference_mode)

    def _execute_eval(inference_mode, override_cfg):
        seg_model = build_seg_inference(
            model,
            loader.dataset,
            cfg,
            eval_key,
            test_cfg_override=override_cfg,
        )
        seg_model.cuda()
        results = multi_gpu_test(
            model=MMDistributedDataParallel(seg_model, device_ids=[torch.cuda.current_device()]),
            data_loader=loader,
            tmpdir=None,
            gpu_collect=True,
            pre_eval=True,
        )
        return results, inference_mode

    try:
        results, used_inference_mode = _execute_eval(requested_inference_mode, test_cfg_override)
    except RuntimeError as exc:
        if validation_mode == "proxy" and requested_inference_mode == "whole" and is_cuda_oom_error(exc):
            logger.warning("[Validation] proxy whole inference OOM; retrying with coarse_slide.")
            torch.cuda.empty_cache()
            results, used_inference_mode = _execute_eval(
                "coarse_slide",
                resolve_proxy_test_cfg("coarse_slide"),
            )
        else:
            raise

    if dist.get_rank() == 0:
        eval_results = loader.dataset.evaluate(results, metric="mIoU", logger=logger)
        payload = build_validation_payload(
            eval_results=eval_results,
            classes=loader.dataset.CLASSES,
            validation_mode=validation_mode,
            used_inference_mode=used_inference_mode,
        )
        payload["dataset_size"] = len(loader.dataset)
        return [payload]
    return [None]


def run_proxy_val(model, loader, eval_key, logger, cfg):
    return _run_seg_eval(
        model,
        loader,
        eval_key,
        logger,
        cfg,
        validation_mode="proxy",
    )


def run_full_val(model, loader, eval_key, logger, cfg):
    return _run_seg_eval(
        model,
        loader,
        eval_key,
        logger,
        cfg,
        validation_mode="full",
    )

@torch.no_grad()
def validate(model, cfg):
    model.eval()
    validation_mode = _validation_mode(cfg)
    all_metrics = {
        "validation_mode": validation_mode,
        "validation_config": {
            "proxy_subset_size": int(cfg.evaluate.get("proxy_subset_size", 300)),
            "proxy_seed": int(cfg.evaluate.get("proxy_seed", cfg.seed)),
            "proxy_inference_mode": str(cfg.evaluate.get("proxy_inference_mode", "whole")),
            "full_eval_top_k": int(cfg.evaluate.get("full_eval_top_k", 3)),
        },
    }
    tasks = cfg.evaluate.task
    for key in tasks:
        dataset = build_eval_dataset(key, validation_mode, cfg)
        loader = build_seg_dataloader(
            dataset,
            workers_per_gpu=_eval_workers_per_gpu(cfg),
        )
        model.apply_found = (key in ["voc", "coco_object"])

        if validation_mode == "proxy":
            metric_list = run_proxy_val(model, loader, cfg.evaluate.get(key), get_logger(), cfg)
        else:
            metric_list = run_full_val(model, loader, cfg.evaluate.get(key), get_logger(), cfg)
        dist.broadcast_object_list(metric_list)
        dist.barrier()

        if metric_list[0] is not None:
            all_metrics[key] = metric_list[0]

    return all_metrics

@dataclass
class DataSelectionResult:
    current_paths: List[str]
    meta: Dict[str, Any]
    scores: List[float]

def prepare_training_subset(
        cfg,
        train_folder: str,
        ann_dir: str,
        logger,
        out_dir: str
) -> DataSelectionResult:
    all_img_paths = sorted([
        os.path.join(train_folder, f)
        for f in os.listdir(train_folder)
        if f.lower().endswith(('.jpg', '.png'))
    ])
    random.seed(cfg.seed)

    pool_n = int(cfg.experiment.get("sample_count", 20000))
    pool_paths = random.sample(all_img_paths, min(pool_n, len(all_img_paths)))

    feature_name = cfg.experiment.get("feature_name", "laplacian")
    feature_kwargs = cfg.experiment.get("feature_kwargs", {}) or {}
    tier_n = int(cfg.train.get("ds_size", 1000))
    subset_strategy = cfg.experiment.get("subset_strategy", "low")
    metric = build_feature(feature_name, feature_kwargs)
    sampler = StratifiedSampler(metric, num_samples_per_tier=tier_n)
    logger.info(
        f"[DataPrep] feature='{feature_name}', pool={len(pool_paths)}, "
        f"tier={tier_n}, strategy='{subset_strategy}'"
    )
    low_path, high_path, mixed_path, random_path, scores = sampler.get_subsets_with_scores(pool_paths, ann_dir)
    if bool(cfg.experiment.get("visualize", True)):
        try:
            plot_metric_distribution(scores, feature_name)
            plot_class_distribution_comparison(
                [low_path, high_path, mixed_path],
                ["Low", "High", "Mixed"],
                ann_dir,
                metrics_name=feature_name
            )
        except Exception as e:
            logger.warning(f"[DataPrep] visualizatoin failed: {e}")
    if subset_strategy == "low":
        current_paths = low_path
    elif subset_strategy == "high":
        current_paths = high_path
    elif subset_strategy == "mixed":
        current_paths = mixed_path
    elif subset_strategy == "random":
        current_paths = random_path
    meta = {
        "seed": int(cfg.seed),
        "feature_name": feature_name,
        "feature_kwargs": feature_kwargs,
        "pool_n": pool_n,
        "tier_n": tier_n,
        "subset_strategy": subset_strategy,
        "counts": {
            "pool": len(pool_paths),
            "low": len(low_path),
            "high": len(high_path),
            "mixed": len(mixed_path),
            "random": len(random_path),
            "train": len(current_paths),
        },
    }
    return DataSelectionResult(
        current_paths=current_paths,
        meta=meta,
        scores=list(scores.values())
    )
                               

# --- 修改后的 main 函数 ---
def main(cfg):
    out_path = cfg.get('output', 'low_sharpness_exp')
    os.makedirs(out_path, exist_ok=True)
    dset_path = cfg.train.get('data')
    train_folder = os.path.join(dset_path, 'images', 'train2017')
    ann_dir = os.path.join(dset_path, 'annotations', 'train2017')
    assert os.path.exists(train_folder), f'Path not found: {train_folder}'
    logger = get_logger(cfg)

    # # 1. 采样与拉普拉斯分数计算 (仅在主进程或 DDP 初始化前执行)
    # all_img_paths = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.lower().endswith(('.jpg', '.png'))]
    # random.seed(cfg.seed)
    # # 随机选 10,000 张进入评分池
    # pool_paths = random.sample(all_img_paths, min(20000, len(all_img_paths)))
    
    # logger.info(f"Calculating Laplacian scores for pool of {len(pool_paths)} images...")
    # sampler = StratifiedSampler(LaplacianSharpness(), num_samples_per_tier=1000)
    # low_path, high_path, mixed_path, scores = sampler.get_subsets_with_scores(pool_paths, ann_dir)
    # plot_metric_distribution(scores, "Laplacian")
    # plot_class_distribution_comparison([low_path, high_path, mixed_path], ["Low", "High", "Mixed"], ann_dir, metrics_name="BGA")
    # current_paths = mixed_path
    # logger.info(f"Alignment complete. Training on {len(current_paths)} images with aligned class distribution.")

    sel = prepare_training_subset(cfg, train_folder, ann_dir, logger, out_dir=out_path)
    current_paths = sel.current_paths
    logger.info(f"Training on {len(current_paths)} images. Selectoin meta: {sel.meta}")

    # 3. 构建自定义 Dataset
    from PIL import Image
    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, file_list, transform=None):
            self.samples = file_list
            self.transform = transform
        def __len__(self): return len(self.samples)
        def __getitem__(self, idx):
            img = Image.open(self.samples[idx]).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, 0

    im_size = cfg.train.get('im_size', 448)
    num_workers = cfg.train.get('num_workers', 4)
    transforms = T.Compose([T.ToTensor(), T.Resize(im_size), T.RandomCrop(im_size), 
                           T.RandomHorizontalFlip(p=0.5), T.ColorJitter(0.5)])
    
    train_dataset = ListDataset(current_paths, transform=transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train.batch_size,
                                               num_workers=num_workers, shuffle=True)

    # 4. 初始化模型与分布式
    if not dist.is_initialized():
        mp.set_start_method("fork", force=True)
        init_dist("pytorch")
    
    rank, world_size = get_dist_info()
    model = build_model(cfg.model, class_names=[''])
    model.load_teachers()

    # 5. 训练
    cudnn.benchmark = True
    do_train(model, cfg.train, {"train": train_loader}, out_path=out_path)
    
    # 6. 验证并保存结果
    logger.info(f"Training finished. Running evaluation...")
    model.found_model = None
    model.vit_encoder = None
    
    results = validate(model, cfg)
    output_result = cfg.output
    # 7. 主进程保存 JSON 结果
    if rank == 0:
        result_name = f"{cfg.experiment.get('subset_strategy', 'low')}.json"
        result_file = os.path.join(out_path, result_name)
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {result_file}")

def parse_args():
    parser = argparse.ArgumentParser(description='CLIP-DINOiser Sharpness Experiment')
    parser.add_argument('config', help='config file path')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    initialize(config_path="configs", version_base=None)
    cfg = compose(config_name=args.config)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    main(cfg)
