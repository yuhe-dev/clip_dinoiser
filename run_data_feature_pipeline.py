# import os
# import json
# import random
# import time
# import torch
# import torch.nn as nn
# import torch.distributed as dist
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
# from PIL import Image
# from omegaconf import OmegaConf
# from hydra import compose, initialize

# # 1. 导入你的 OOP 模块
# from feature_utils.data_feature.implementations.quality import LaplacianSharpness
# from feature_utils.data_feature.sampler import StratifiedSampler
# from feature_utils.visualizer import plot_metric_distribution, plot_class_distribution_comparison

# # 2. 导入模型与训练组件
# import torchvision.transforms as T
# from models import build_model
# from scheduler import MultiStepLR
# from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_seg_inference
# from mmcv.runner import get_dist_info, init_dist
# from mmcv.parallel import MMDistributedDataParallel
# from mmseg.apis import multi_gpu_test

# # --- 强制注册 mmsegmentation 算子 ---
# try:
#     import mmseg.datasets.pipelines as mmseg_pipelines
#     from mmseg.datasets import PIPELINES
#     # 这一行是关键：它会触发整个 pipelines 文件夹下的所有算子注册
#     from mmseg.datasets.pipelines import Compose
#     print(f"Successfully registered {len(PIPELINES)} mmseg pipelines.")
# except Exception as e:
#     print(f"Warning: Failed to force-register mmseg pipelines: {e}")
# # ----------------------------------

# # --- 辅助类：数据读取 ---
# class ListDataset(torch.utils.data.Dataset):
#     def __init__(self, file_list, transform=None):
#         self.samples = file_list
#         self.transform = transform
#     def __len__(self): return len(self.samples)
#     def __getitem__(self, idx):
#         img = Image.open(self.samples[idx]).convert('RGB')
#         if self.transform: img = self.transform(img)
#         return img, 0

# # ==========================================
# # 核心逻辑 A: 结果对比可视化引擎 (Plot 2 & 3)
# # ==========================================
# def visualize_experiment_results(json_files):
#     """
#     json_files: {'Low': 'path/to/low.json', ...}
#     """
#     all_summary = []
#     all_per_class = []

#     for strategy, path in json_files.items():
#         if not os.path.exists(path): continue
#         with open(path, 'r') as f:
#             data = json.load(f)
#             # 假设只关注 coco_stuff 任务
#             task_data = data.get('coco_stuff', {})
            
#             # 汇总总指标
#             summary = task_data.get('summary', {})
#             summary['Strategy'] = strategy
#             all_summary.append(summary)
            
#             # 汇总逐类指标
#             for cls, metrics in task_data.get('per_class', {}).items():
#                 all_per_class.append({
#                     'Strategy': strategy,
#                     'Class': cls,
#                     'IoU': metrics['IoU']
#                 })

#     df_s = pd.DataFrame(all_summary)
#     df_c = pd.DataFrame(all_per_class)

#     # --- Plot 2: 总体指标对比 ---
#     plt.figure(figsize=(10, 6))
#     df_melted = df_s.melt(id_vars='Strategy', value_vars=['mIoU'])
#     ax = sns.barplot(data=df_melted, x='variable', y='value', hue='Strategy', palette='viridis')
#     plt.title("Plot 2: Global mIoU Comparison", fontsize=14)
#     plt.ylabel("Score (%)")
#     for p in ax.patches:
#         ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
#     plt.savefig("plot_2_global_comparison.png")

#     # --- Plot 3: 性能差距最大的 Top 20 类别 (Lollipop Chart) ---
#     # 计算每个类的 IoU 极差
#     pivot_df = df_c.pivot(index='Class', columns='Strategy', values='IoU')
#     pivot_df['Gap'] = pivot_df.max(axis=1) - pivot_df.min(axis=1)
#     top_gap_classes = pivot_df.sort_values('Gap', ascending=False).head(20).index
    
#     df_top = df_c[df_c['Class'].isin(top_gap_classes)]
#     plt.figure(figsize=(12, 8))
#     sns.pointplot(data=df_top, y='Class', x='IoU', hue='Strategy', join=False, palette='bright', dodge=0.3)
#     plt.title("Plot 3: Per-class IoU Gaps (Top 20 Most Sensitive Classes)", fontsize=14)
#     plt.grid(axis='x', linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plt.savefig("plot_3_class_gaps.png")
#     print("Results plots generated: plot_2 and plot_3.")

# # ==========================================
# # 核心逻辑 B: 单次训练与验证循环
# # ==========================================
# def run_training_and_eval(strategy_name, paths, cfg):
#     rank, _ = get_dist_info()
    
#     # 1. 准备 DataLoader
#     im_size = cfg.train.get('im_size', 448)
#     trans = T.Compose([T.ToTensor(), T.Resize(im_size), T.RandomCrop(im_size), T.RandomHorizontalFlip(), T.ColorJitter(0.5)])
#     loader = torch.utils.data.DataLoader(ListDataset(paths, trans), batch_size=cfg.train.batch_size, num_workers=4, shuffle=True)

#     # 2. 初始化模型
#     model = build_model(cfg.model, class_names=[''])
#     model.load_teachers()
#     model.to("cuda")

#     # 3. 训练 (极简版实现，确保显存回收)
#     optimizer = torch.optim.AdamW([{'params': model.obj_proj.parameters()}, {'params': model.bkg_decoder.parameters(), 'lr': cfg.train.found_lr}], lr=cfg.train.corr_lr)
#     criterion = nn.BCEWithLogitsLoss()
    
#     for epoch in range(cfg.train.get('epochs', 5)): # 默认跑5个epoch验证
#         for data in tqdm(loader, desc=f"Training {strategy_name} Epoch {epoch}", disable=(rank!=0)):
#             model.train()
#             inputs = data[0].to("cuda")
#             optimizer.zero_grad()
#             preds_bkg, pred_corrs, _ = model.forward_pass(inputs)
#             with torch.no_grad():
#                 found_mask = (model.get_found_preds(inputs, resize=preds_bkg.shape[-2:]) > 0.5).float()
#                 dino_mask = (model.get_dino_corrs(inputs).detach() > 0).float()
#             loss = criterion(pred_corrs.flatten(-2,-1), dino_mask.flatten(-2,-1)) + criterion(preds_bkg.flatten(-2,-1), found_mask.flatten(-2,-1))
#             loss.backward()
#             optimizer.step()

#     # 4. 验证
#     model.eval()
#     final_results = {}

#     import mmseg.datasets.pipelines

#     for task_key in cfg.evaluate.task:
#         val_loader = build_seg_dataloader(build_seg_dataset(cfg.evaluate.get(task_key)))
#         model.clip_backbone.decode_head.update_vocab(val_loader.dataset.CLASSES)
#         seg_model = MMDistributedDataParallel(build_seg_inference(model, val_loader.dataset, cfg, cfg.evaluate.get(task_key)).cuda(), device_ids=[torch.cuda.current_device()])
        
#         raw_res = multi_gpu_test(seg_model, val_loader, gpu_collect=True, pre_eval=True)
#         if rank == 0:
#             eval_res = val_loader.dataset.evaluate(raw_res, metric="mIoU")
#             final_results[task_key] = {
#                 "summary": {"mIoU": float(eval_res['mIoU']*100), "mAcc": float(eval_res['mAcc']*100)},
#                 "per_class": {name: {"IoU": float(eval_res.get(f'IoU.{name}', 0)*100)} for name in val_loader.dataset.CLASSES}
#             }
    
#     # 5. 显存清理
#     del model, optimizer, loader
#     torch.cuda.empty_cache()
#     return final_results

# # ==========================================
# # 核心逻辑 C: 主流水线
# # ==========================================
# def main():
#     initialize(config_path="configs", version_base=None)
#     cfg = compose(config_name="laplacian.yaml")
#     init_dist("pytorch")
#     rank, _ = get_dist_info()

#     # --- Step 1: 采样 (仅在 Rank 0 执行并分发) ---
#     train_folder = "data/coco_stuff164k/images/train2017"
#     ann_dir = "data/coco_stuff164k/annotations/train2017"
    
#     if rank == 0:
#         all_imgs = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.jpg')]
#         pool = random.sample(all_imgs, 2000)
        
#         # 使用 OOP 采样器
#         metric_tool = LaplacianSharpness()
#         sampler = StratifiedSampler(metric_tool, num_samples_per_tier=1000)
#         low_p, high_p, mixed_p, scores_dict = sampler.get_subsets_with_scores(pool, ann_dir)
        
#         # 【图 1】 分数分布
#         plot_metric_distribution(scores_dict, "Laplacian")
#         # 【图 4】 类别对齐检查
#         plot_class_distribution_comparison([low_p, high_p, mixed_p], ["Low", "High", "Mixed"], ann_dir)
        
#         data_to_share = [low_p, high_p, mixed_p]
#     else:
#         data_to_share = [None, None, None]

#     dist.broadcast_object_list(data_to_share, src=0)
#     strategies = {"Low": data_to_share[0], "High": data_to_share[1], "Mixed": data_to_share[2]}

#     # --- Step 2: 顺序实验 ---
#     json_paths = {}
#     for name, paths in strategies.items():
#         results = run_training_and_eval(name, paths, cfg)
#         if rank == 0:
#             path = f"{name.lower()}.json"
#             with open(path, 'w') as f:
#                 json.dump(results, f, indent=4)
#             json_paths[name] = path

#     # --- Step 3: 最终对比图 ---
#     if rank == 0:
#         visualize_experiment_results(json_paths)
#         print("\nPipeline execution complete. All 4 plots generated.")

# if __name__ == "__main__":
#     main()

import os
import json
import random
import time
import cv2
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from hydra import compose, initialize

# --- 1. 严格同步 laplacian_v2.py 的 mmseg 相关导入 ---
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist
from mmseg.apis import multi_gpu_test
from helpers.logger import get_logger
from models import build_model
from scheduler import MultiStepLR
from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_seg_inference

# --- 2. 你的 OOP 模块 ---
from feature_utils.data_feature.implementations.quality import LaplacianSharpness
from feature_utils.data_feature.sampler import StratifiedSampler
from feature_utils.visualizer import plot_metric_distribution, plot_class_distribution_comparison

import torchvision.transforms as T
from mmseg.datasets.builder import PIPELINES
import torch

# --- 强力注入：手动补齐 mmseg 注册表中的缺失算子 ---

@PIPELINES.register_module(force=True)
class ToRGB:
    def __init__(self): pass
    def __call__(self, results): return results

@PIPELINES.register_module(force=True)
class ImageToTensorV2:
    def __init__(self, keys):
        self.keys = keys
    def __call__(self, results):
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # 转换为 Tensor: HWC -> CHW
            results[key] = torch.from_numpy(img.transpose(2, 0, 1))
        return results

@PIPELINES.register_module(force=True)
class Collect:
    def __init__(self, keys, meta_keys=()):
        self.keys = keys
        self.meta_keys = meta_keys
    def __call__(self, results):
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = img_meta
        for key in self.keys:
            data[key] = results[key]
        return data

# ----------------------------------------------

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, transform=None):
        self.samples = file_list
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, 0

# ==========================================
# 核心逻辑 A: 结果对比可视化引擎 (Plot 2 & 3)
# ==========================================
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

# ==========================================
# 核心逻辑 B: 评估函数 (完全同步 laplacian_v2.py)
# ==========================================
def run_val(model, loader, eval_key, logger, cfg):
    model.clip_backbone.decode_head.update_vocab(loader.dataset.CLASSES)
    seg_model = build_seg_inference(model, loader.dataset, cfg, eval_key)
    seg_model.cuda()
    
    results = multi_gpu_test(
        model=MMDistributedDataParallel(seg_model, device_ids=[torch.cuda.current_device()]),
        data_loader=loader, tmpdir=None, gpu_collect=True, pre_eval=True
    )

    if dist.get_rank() == 0:
        eval_results = loader.dataset.evaluate(results, metric="mIoU", logger=logger)
        metrics_dict = {
            "summary": {
                "mIoU": float(eval_results.get('mIoU', 0) * 100),
                "mAcc": float(eval_results.get('mAcc', 0) * 100),
                "aAcc": float(eval_results.get('aAcc', 0) * 100)
            },
            "per_class": {}
        }
        for i, class_name in enumerate(loader.dataset.CLASSES):
            metrics_dict["per_class"][class_name] = {
                "IoU": float(eval_results.get(f'IoU.{class_name}', 0) * 100),
                "Acc": float(eval_results.get(f'Acc.{class_name}', 0) * 100)
            }
        return [metrics_dict]
    return [None]

@torch.no_grad()
def validate(model, cfg):
    model.eval()
    all_metrics = {}
    tasks = cfg.evaluate.task
    for key in tasks:
        # 这里会调用你 stuff.py 里的配置，如果 import 链对齐了，ToRGB 就不会报错
        loader = build_seg_dataloader(build_seg_dataset(cfg.evaluate.get(key)))
        model.apply_found = (key in ["voc", "coco_object"])
        metric_list = run_val(model, loader, cfg.evaluate.get(key), get_logger(), cfg)
        dist.broadcast_object_list(metric_list)
        dist.barrier()
        if metric_list[0] is not None:
            all_metrics[key] = metric_list[0]
    return all_metrics

# ==========================================
# 核心逻辑 C: 训练逻辑
# ==========================================
def run_training_loop(strategy_name, paths, cfg):
    rank, _ = get_dist_info()
    

    im_size = cfg.train.get('im_size', 448)
    transforms = T.Compose([T.ToTensor(), T.Resize(im_size), T.RandomCrop(im_size), T.RandomHorizontalFlip(p=0.5), T.ColorJitter(0.5)])
    train_loader = torch.utils.data.DataLoader(ListDataset(paths, transform=transforms), batch_size=cfg.train.batch_size, num_workers=4, shuffle=True)

    # 模型初始化
    model = build_model(cfg.model, class_names=[''])
    model.load_teachers()
    model.to("cuda")

    # 优化器
    optimizer = torch.optim.AdamW([{'params': model.obj_proj.parameters()}, {'params': model.bkg_decoder.parameters(), 'lr': cfg.train.get('found_lr')}], lr=cfg.train.get('corr_lr'))
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    scheduler = MultiStepLR(optimizer, cfg.train.get('milestones'), gamma=cfg.train.get("step_lr_gamma"), warmup=0)

    # 简易训练循环 (可根据需要增加 Epoch)
    for epoch in range(cfg.train.epochs):
        tbar = tqdm(train_loader, disable=(rank != 0))
        for data in tbar:
            model.train()
            inputs = data[0].to("cuda")
            optimizer.zero_grad()
            preds_bkg, pred_corrs, _ = model.forward_pass(inputs)
            with torch.no_grad():
                found_pred = (model.get_found_preds(inputs, resize=preds_bkg.shape[-2:]) > 0.5).float()
                dino_corrs = (model.get_dino_corrs(inputs).detach() > 0).float()
            loss = criterion(pred_corrs.flatten(-2, -1), dino_corrs.flatten(-2, -1)) + criterion(preds_bkg.flatten(-2, -1), found_pred.flatten(-2, -1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            if rank == 0: tbar.set_description(f"[{strategy_name}] Epoch {epoch} Loss: {loss.item():.4f}")

    # 评估阶段
    results = validate(model, cfg)
    
    # 释放显存
    del model, optimizer, train_loader
    torch.cuda.empty_cache()
    return results

# ==========================================
# 核心逻辑 D: 主流程
# ==========================================
def main():
    initialize(config_path="configs", version_base=None)
    cfg = compose(config_name="laplacian.yaml")
    
    if not dist.is_initialized():
        init_dist("pytorch")
    
    rank, _ = get_dist_info()
    cudnn.benchmark = True

    # 采样逻辑
    train_folder = "data/coco_stuff164k/images/train2017"
    ann_dir = "data/coco_stuff164k/annotations/train2017"
    
    if rank == 0:
        all_paths = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.jpg')]
        pool = random.sample(all_paths, 500)
        sampler = StratifiedSampler(LaplacianSharpness(), num_samples_per_tier=100)
        low_p, high_p, mixed_p, scores = sampler.get_subsets_with_scores(pool, ann_dir)
        
        plot_metric_distribution(scores, "Laplacian")
        plot_class_distribution_comparison([low_p, high_p, mixed_p], ["Low", "High", "Mixed"], ann_dir)
        path_data = [low_p, high_p, mixed_p]
    else:
        path_data = [None, None, None]

    dist.broadcast_object_list(path_data, src=0)
    strategies = {"Low": path_data[0], "High": path_data[1], "Mixed": path_data[2]}

    json_paths = {}
    for name, paths in strategies.items():
        results = run_training_loop(name, paths, cfg)
        if rank == 0:
            save_path = f"new_{name.lower()}.json"
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=4)
            json_paths[name] = save_path

    if rank == 0:
        visualize_experiment_results(json_paths)
        print("Pipeline Complete. All plots generated.")

if __name__ == "__main__":
    main()