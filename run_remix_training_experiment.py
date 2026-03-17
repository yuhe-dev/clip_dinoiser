from __future__ import annotations

import argparse
import json
import os
import random
import sys

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
    do_train(model, cfg.train, {"train": train_loader}, out_path=cfg.output)
    logger.info("Training finished. Running evaluation...")
    model.found_model = None
    model.vit_encoder = None
    results = validate(model, cfg)

    if rank == 0:
        result_path = os.path.join(cfg.output, args.result_name)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        manifest_path = os.path.join(cfg.output, f"{os.path.splitext(args.result_name)[0]}_result_entry.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "candidate_id": manifest.candidate_id,
                    "subset_manifest": os.path.abspath(args.subset_manifest),
                    "result_path": result_path,
                    "seed": int(cfg.seed),
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
