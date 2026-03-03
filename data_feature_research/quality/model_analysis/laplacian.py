import sys
import os

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上跳三级目录，到达 clip_dinoiser 根目录
# 因为你的路径是 data_feature_research/quality/model_analysis/laplacian.py
root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from datasets import transforms

import argparse
import os
import random
import time
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

from helpers.logger import get_logger
from models import build_model
from scheduler import MultiStepLR
from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_seg_inference
import cv2
from scipy.stats import rankdata

# --- 新增：拉普拉斯分数计算函数 ---
def get_sharpness_score(path):
    try:
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except:
        return 0

# --- 新增：结果可视化函数 ---
def plot_comparison_results(results_dict, out_path):
    import matplotlib.pyplot as plt
    labels = list(results_dict.keys())
    # 假设我们关注 coco_object_miou
    tasks = list(results_dict[labels[0]].keys())
    
    for task in tasks:
        scores = [results_dict[label][task] for label in labels]
        plt.figure(figsize=(10, 6))
        plt.bar(labels, scores, color=['red', 'orange', 'green'])
        plt.title(f'Comparison of {task} across Sharpness Tiers')
        plt.ylabel('mIoU (%)')
        plt.savefig(os.path.join(out_path, f'comparison_{task.replace("/", "_")}.png'))
        plt.close()

def get_model_dict(model):
    new_check = {}
    new_check['obj_proj.bias'] = model.state_dict()['obj_proj.bias'].cpu()
    new_check['obj_proj.weight'] = model.state_dict()['obj_proj.weight'].cpu()
    new_check['bkg_decoder.bias'] = model.state_dict()['bkg_decoder.bias'].cpu()
    new_check['bkg_decoder.weight'] = model.state_dict()['bkg_decoder.weight'].cpu()
    return new_check


def get_criterion(cfg):
    if cfg.get('loss') == 'CE':
        return torch.nn.BCEWithLogitsLoss(reduction='mean')
    else:
        raise NotImplementedError


def do_train(model, train_cfg, loaders, out_path):
    timestamp = time.time()
    date_time = datetime.fromtimestamp(timestamp)
    str_date_time = date_time.strftime("%d%m%Y-%H%M%S")

    ch_path = os.path.join(out_path, str_date_time)
    os.mkdir(ch_path)
    model.to("cuda")
    epochs = train_cfg.get("epochs", 100)
    criterion = get_criterion(train_cfg)
    optimizer = torch.optim.AdamW([{'params': model.obj_proj.parameters()},
                                   {'params': model.bkg_decoder.parameters(), 'lr': train_cfg.get('found_lr')}],
                                  lr=train_cfg.get('corr_lr'))
    scheduler = MultiStepLR(optimizer, train_cfg.get('milestones'), gamma=train_cfg.get("step_lr_gamma"), warmup=0)

    for epoch in range(epochs):
        tbar = tqdm(enumerate(loaders['train'], 0))
        for i, data in tbar:
            model.bkg_decoder.train()
            model.obj_proj.train()
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
            optimizer.step()

            tbar.set_description(f"{epoch}: {i} | {loss.item()}")
            scheduler.step()

    # save checkpoint
    model.found_model = None
    model.vit_encoder = None
    torch.save({
        'epoch': epoch, # type:ignore
        'model_state_dict': get_model_dict(model),
    }, os.path.join(ch_path, 'last.pt'))


@torch.no_grad()
def validate(model, cfg):
    model.eval()
    logger = get_logger()
    ret = {}
    tasks = cfg.evaluate.task

    for key in tasks:
        loader = build_seg_dataloader(build_seg_dataset(cfg.evaluate.get(key)))
        model.apply_found = False
        if key in ["voc", "coco_object"]:
            model.apply_found = True
        metric = run_val(model, loader, cfg.evaluate.get(key), logger, cfg)
        dist.broadcast_object_list(metric)
        torch.cuda.empty_cache()
        dist.barrier()
        ret[f"val/{key}_miou"] = metric[0]["mIoU"] * 100 # type:ignore
    logger.info(ret)
    return ret


def run_val(model, loader, eval_key, logger, cfg):
    model.clip_backbone.decode_head.update_vocab(loader.dataset.CLASSES)

    seg_model = build_seg_inference(
        model,
        loader.dataset,
        cfg,
        eval_key)
    seg_model.cuda()
    model.device = 'cuda'

    results = multi_gpu_test(
        model=MMDistributedDataParallel(seg_model, device_ids=[torch.cuda.current_device()]),
        data_loader=loader,
        tmpdir=None,
        gpu_collect=True,
        efficient_test=False,
        pre_eval=True,
        format_only=False,
    )

    if dist.get_rank() == 0:
        metric = [loader.dataset.evaluate(results, metric="mIoU", logger=logger)]
    else:
        metric = [None]
    return metric
def main(cfg):
    # 1. 初始化路径与配置
    out_path = cfg.get('output', 'exp_results')
    os.makedirs(out_path, exist_ok=True)
    dset_path = cfg.train.get('data')
    train_folder = os.path.join(dset_path, 'images', 'train2017')
    logger = get_logger(cfg)

    # 2. 采样与分数预计算
    all_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.lower().endswith(('.jpg', '.png'))]
    random.seed(cfg.seed)
    sampled_files = random.sample(all_files, min(20000, len(all_files)))
    
    logger.info("Step 1: Calculating Laplacian scores for 20,000 images...")
    scores = []
    for f in tqdm(sampled_files):
        scores.append(get_sharpness_score(f))
    
    # 3. 数据分层 (Stratification)
    scores = np.array(scores)
    sorted_indices = np.argsort(scores) # 从小到大排
    
    n = len(sorted_indices)
    tiers = {
        "Low_Sharpness": sorted_indices[:n//3],
        "Mid_Sharpness": sorted_indices[n//3 : 2*n//3],
        "High_Sharpness": sorted_indices[2*n//3:]
    }

    # 存储最终对比结果
    final_metrics = {}

    # 4. 循环训练三个 Tiers
    for tier_name, indices in tiers.items():
        logger.info(f"\n>>> Starting Training for Tier: {tier_name} (Size: {len(indices)})")
        
        # 构建当前 Tier 的 Dataset
        tier_files = [sampled_files[i] for i in indices]
        
        class TierDataset(torch.utils.data.Dataset):
            def __init__(self, file_list, transform=None):
                self.files = file_list
                self.transform = transform
            def __len__(self): return len(self.files)
            def __getitem__(self, idx):
                img = cv2.imread(self.files[idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                from PIL import Image
                img = Image.fromarray(img)
                if self.transform: img = self.transform(img)
                return img, 0

        im_size = cfg.train.get('im_size', 448)
        transforms = T.Compose([T.ToTensor(), T.Resize(im_size), T.RandomCrop(im_size), 
                               T.RandomHorizontalFlip(), T.ColorJitter(0.5)])
        train_dataset = TierDataset(tier_files, transform=transforms)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train.batch_size, 
                                                   num_workers=4, shuffle=True)

        # 实例化模型
        model = build_model(cfg.model, class_names=[''])
        model.load_teachers()
        
        # 训练
        tier_out_path = os.path.join(out_path, tier_name)
        os.makedirs(tier_out_path, exist_ok=True)
        do_train(model, cfg.train, {"train": train_loader}, out_path=tier_out_path)
        
        # 验证并记录指标
        logger.info(f"Evaluating {tier_name}...")
        model.found_model = None # 释放显存
        model.vit_encoder = None
        metrics = validate(model, cfg) # 修改 validate 让其返回结果字典
        final_metrics[tier_name] = metrics

    # 5. 输出文本报告与可视化图表
    with open(os.path.join(out_path, "summary_report.txt"), "w") as f:
        f.write("CLIP-DINOiser Sharpness Experiment Report\n")
        f.write("="*40 + "\n")
        for tier, m in final_metrics.items():
            f.write(f"Tier: {tier} | Metrics: {m}\n")
    
    plot_comparison_results(final_metrics, out_path)
    logger.info(f"All experiments finished. Results saved in {out_path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description='CLIP-DINOiser training procedure')
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