import os, random, json
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from hydra import compose, initialize
from mmcv.runner import init_dist, get_dist_info, set_random_seed
from mmcv.parallel import MMDistributedDataParallel
from mmseg.apis import multi_gpu_test

from helpers.logger import get_logger
from models import build_model
from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_seg_inference
from datasets import transforms

def stable_sample_image_paths(train_folder, seed, k):
    all_img_paths = sorted([
        os.path.join(train_folder, f)
        for f in os.listdir(train_folder)
        if f.lower().endswith(('.jpg', '.png'))
    ])
    rng = random.Random(seed)
    return rng.sample(all_img_paths, min(k, len(all_img_paths)))


def subset_dataset_by_paths(dataset, keep_img_paths):
    keep = set(os.path.basename(p) for p in keep_img_paths)

    if hasattr(dataset, "img_infos"):
        infos = dataset.img_infos
        key = "img_infos"
    elif hasattr(dataset, "data_infos"):
        infos = dataset.data_infos
        key = "data_infos"
    else:
        raise RuntimeError("Dataset has no img_infos/data_infos.")

    new_infos = []
    for info in infos:
        fn = info.get("filename", info.get("img_path", None))
        if fn is None:
            continue
        if os.path.basename(fn) in keep:
            new_infos.append(info)

    setattr(dataset, key, new_infos)
    return dataset


def per_image_miou(results, dataset, thing_only=False):
    C = len(dataset.CLASSES)
    allowed = np.ones(C, dtype=bool)
    if thing_only:
        allowed[:] = False
        allowed[:min(80, C)] = True

    def get_id(i):
        if hasattr(dataset, "img_infos"):
            return dataset.img_infos[i].get("filename", str(i))
        if hasattr(dataset, "data_infos"):
            d = dataset.data_infos[i]
            return d.get("filename", d.get("img_path", str(i)))
        return str(i)

    out = []
    for i, (inter, union, pred, gt) in enumerate(results):
        inter = inter.detach().cpu().numpy().astype(np.float64)
        union = union.detach().cpu().numpy().astype(np.float64)
        gt = gt.detach().cpu().numpy().astype(np.float64)

        present = (gt > 0) & allowed
        miou = float((inter[present] / (union[present] + 1e-10)).mean()) if present.any() else 0.0
        out.append({"img": get_id(i), "mIoU": miou, "difficulty": 1.0 - miou})
    return out


@torch.no_grad()
def offline_eval(cfg):
    logger = get_logger(cfg)
    rank, world = get_dist_info()

    # 1) build dataset/loader (use COCO-Stuff train config!)
    key = "coco_stuff"  # 你也可以从 cfg 里读
    seg_cfg = cfg.evaluate.get(key)
    dataset = build_seg_dataset(seg_cfg)

    # 2) sample 20k from train2017 images
    dset_path = cfg.train.get("data")
    train_folder = os.path.join(dset_path, "images", "train2017")
    keep_paths = stable_sample_image_paths(train_folder, seed=int(cfg.seed), k=int(cfg.experiment.get("sample_count", 20)))

    dataset = subset_dataset_by_paths(dataset, keep_paths)

    # debug: inspect first record
    info0 = dataset.img_infos[0] if hasattr(dataset, "img_infos") else dataset.data_infos[0]
    print("[DEBUG] dataset len:", len(dataset))
    print("[DEBUG] dataset first info keys:", list(info0.keys()))
    print("[DEBUG] dataset first info:", info0)

    # debug: see if val2017 in filenames
    def _get_fn(info):
        if "filename" in info: return info["filename"]
        if "img_path" in info: return info["img_path"]
        if "img_info" in info and "filename" in info["img_info"]: return info["img_info"]["filename"]
        return None

    print("[DEBUG] first filename:", _get_fn(info0))

    loader = build_seg_dataloader(dataset)

    logger.info(f"[OfflineEval] subset size = {len(dataset)}")

    # 3) load model from checkpoints/last.pt
    model = build_model(cfg.model, class_names=dataset.CLASSES)
    model.apply_found = False
    ckpt_path = cfg.get("ckpt_path", "checkpoints/last.pt")
    state = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.cuda()
    model.eval()

    # 4) build seg inference wrapper + ddp wrapper
    seg_model = build_seg_inference(model, dataset, cfg, seg_cfg)
    seg_model.cuda().eval()
    mmddp_model = MMDistributedDataParallel(seg_model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
    mmddp_model.eval()

    # 5) run inference pre_eval stats
    results = multi_gpu_test(
        model=mmddp_model,
        data_loader=loader,
        tmpdir=None,
        gpu_collect=True,
        pre_eval=True,
        format_only=False,
    )

    # 6) rank0: compute per-image + save jsonl
    if dist.get_rank() == 0:
        rows = per_image_miou(results, dataset, thing_only=False)
        out_dir = cfg.get("output", "offline_scores")
        os.makedirs(out_dir, exist_ok=True)

        out_file = os.path.join(out_dir, f"empirical_miou_{key}_seed{cfg.seed}_n{len(rows)}.jsonl")
        with open(out_file, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        logger.info(f"[OfflineEval] saved per-image scores to {out_file}")

        # optional: also compute dataset-level mIoU using mmseg's evaluate
        metric = dataset.evaluate(results, metric="mIoU", logger=logger)
        logger.info(f"[OfflineEval] dataset mIoU = {metric['mIoU']*100:.2f}%")

    dist.barrier()


def main(cfg):
    mp.set_start_method("fork", force=True)
    init_dist("pytorch")
    dist.barrier()
    set_random_seed(cfg.seed, use_rank_shift=True)
    cudnn.benchmark = True
    offline_eval(cfg)
    dist.barrier()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    initialize(config_path="configs", version_base=None)
    cfg = compose(config_name=args.config)
    main(cfg)