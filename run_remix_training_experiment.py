from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import random
import sys
import time

import numpy as np

if __package__ in {None, ""}:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from slice_remix.manifests import load_subset_manifest
else:
    from .slice_remix.manifests import load_subset_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a manifest-driven remix training/evaluation experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--subset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-name", required=True)
    parser.add_argument("--pool-image-root")
    parser.add_argument("--seed", type=int)
    return parser


def _progress(message: str) -> None:
    print(f"[remix_training_experiment] {message}", file=sys.stderr, flush=True)


def build_timing_summary(
    *,
    train_seconds: float,
    eval_seconds: float,
    total_seconds: float,
    subset_size: int,
    started_at: str,
    finished_at: str,
) -> dict[str, object]:
    return {
        "subset_size": int(subset_size),
        "started_at": started_at,
        "finished_at": finished_at,
        "train_seconds": round(float(train_seconds), 3),
        "eval_seconds": round(float(eval_seconds), 3),
        "total_seconds": round(float(total_seconds), 3),
    }


def build_result_entry_filename(candidate_id: str) -> str:
    return f"{candidate_id}_result_entry.json"


def run(args: argparse.Namespace) -> int:
    from PIL import Image
    import torch
    import torch.backends.cudnn as cudnn
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torchvision.transforms as T
    from hydra import compose, initialize
    from mmcv.runner import get_dist_info, init_dist

    from feature_experiment_pipeline import do_train, validate
    from helpers.logger import get_logger
    from models import build_model

    manifest = load_subset_manifest(
        os.path.abspath(args.subset_manifest),
        pool_image_root=os.path.abspath(args.pool_image_root) if args.pool_image_root else None,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    total_start = time.perf_counter()
    _progress(f"loaded manifest candidate_id={manifest.candidate_id} subset_size={len(manifest.sample_paths)}")

    initialize(config_path="configs", version_base=None)
    cfg = compose(config_name=args.config)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    cfg.output = os.path.abspath(args.output_dir)

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    random.seed(int(cfg.seed))
    os.makedirs(cfg.output, exist_ok=True)
    logger = get_logger(cfg)

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, file_list, transform=None):
            self.samples = list(file_list)
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img = Image.open(self.samples[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, 0

    im_size = cfg.train.get("im_size", 448)
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize(im_size),
            T.RandomCrop(im_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.5),
        ]
    )
    train_dataset = ListDataset(manifest.sample_paths, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.get("num_workers", 4),
        shuffle=True,
    )

    if not dist.is_initialized():
        mp.set_start_method("fork", force=True)
        init_dist("pytorch")

    rank, _ = get_dist_info()
    model = build_model(cfg.model, class_names=[""])
    model.load_teachers()

    cudnn.benchmark = True
    train_start = time.perf_counter()
    _progress("starting training")
    do_train(model, cfg.train, {"train": train_loader}, out_path=cfg.output)
    train_seconds = time.perf_counter() - train_start
    _progress(f"training finished train_seconds={train_seconds:.3f}")
    logger.info("Training finished. Running evaluation...")
    model.found_model = None
    model.vit_encoder = None
    eval_start = time.perf_counter()
    _progress("starting evaluation")
    results = validate(model, cfg)
    eval_seconds = time.perf_counter() - eval_start
    total_seconds = time.perf_counter() - total_start
    finished_at = datetime.now(timezone.utc).isoformat()
    timing = build_timing_summary(
        train_seconds=train_seconds,
        eval_seconds=eval_seconds,
        total_seconds=total_seconds,
        subset_size=len(manifest.sample_paths),
        started_at=started_at,
        finished_at=finished_at,
    )
    _progress(
        "finished candidate_id="
        f"{manifest.candidate_id} train_seconds={timing['train_seconds']:.3f} "
        f"eval_seconds={timing['eval_seconds']:.3f} total_seconds={timing['total_seconds']:.3f}"
    )

    if rank == 0:
        if isinstance(results, dict):
            results = dict(results)
            results["timing"] = timing
        result_path = os.path.join(cfg.output, args.result_name)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        manifest_path = os.path.join(cfg.output, build_result_entry_filename(manifest.candidate_id))
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "candidate_id": manifest.candidate_id,
                    "subset_manifest": os.path.abspath(args.subset_manifest),
                    "result_path": result_path,
                    "seed": int(cfg.seed),
                    "timing": timing,
                },
                f,
                indent=2,
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
