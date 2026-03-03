# ---------------------------------------------------------------------------------------------------
# CLIP-DINOiser: Stratified Training on Low-Sharpness Subset
# ---------------------------------------------------------------------------------------------------
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

from helpers.logger import get_logger
from models import build_model
from scheduler import MultiStepLR
from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_seg_inference

# --- 新增：拉普拉斯分数计算 ---
def get_sharpness_score(path):
    try:
        image = cv2.imread(path)
        if image is None: return 0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except Exception:
        return 0

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
    os.makedirs(ch_path, exist_ok=True)
    model.to("cuda")
    epochs = train_cfg.get("epochs", 100)
    criterion = get_criterion(train_cfg)
    optimizer = torch.optim.AdamW([{'params': model.obj_proj.parameters()},
                                   {'params': model.bkg_decoder.parameters(), 'lr': train_cfg.get('found_lr')}],
                                  lr=train_cfg.get('corr_lr'))
    scheduler = MultiStepLR(optimizer, train_cfg.get('milestones'), gamma=train_cfg.get("step_lr_gamma"), warmup=0)

    for epoch in range(epochs):
        tbar = tqdm(enumerate(loaders['train'], 0), disable=(get_dist_info()[0] != 0))
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

            if get_dist_info()[0] == 0:
                tbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            scheduler.step()

    # save checkpoint
    if get_dist_info()[0] == 0:
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
        if metric[0] is not None:
            ret[f"val/{key}_miou"] = metric[0]["mIoU"] * 100 
    logger.info(ret)
    return ret # 修改：返回结果字典


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

# --- 修改后的 main 函数 ---
def main(cfg):
    out_path = cfg.get('output', 'low_sharpness_exp')
    os.makedirs(out_path, exist_ok=True)
    dset_path = cfg.train.get('data')
    train_folder = os.path.join(dset_path, 'images', 'train2017')
    assert os.path.exists(train_folder), f'Path not found: {train_folder}'
    logger = get_logger(cfg)

    # 1. 采样与拉普拉斯分数计算 (仅在主进程或 DDP 初始化前执行)
    all_img_paths = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.lower().endswith(('.jpg', '.png'))]
    random.seed(cfg.seed)
    # 随机选 10,000 张进入评分池
    pool_paths = random.sample(all_img_paths, min(10000, len(all_img_paths)))
    
    logger.info(f"Calculating Laplacian scores for pool of {len(pool_paths)} images...")
    path_score_pairs = []
    for p in tqdm(pool_paths):
        path_score_pairs.append((p, get_sharpness_score(p)))
    
    # 2. 排序并选择最低的 1000 张
    path_score_pairs.sort(key=lambda x: x[1])
    low_sharpness_paths = [x[0] for x in path_score_pairs[:1000]]
    logger.info(f"Selected 1000 lowest sharpness images. Score range: {path_score_pairs[0][1]:.2f} to {path_score_pairs[999][1]:.2f}")

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
    
    train_dataset = ListDataset(low_sharpness_paths, transform=transforms)
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
    
    # 7. 主进程保存 JSON 结果
    if rank == 0:
        result_file = os.path.join(out_path, 'metrics_summary.json')
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