#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

from research_harness.supervised_probe import run_supervised_probe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a modern PyTorch/mmseg supervised segmentation probe learner."
    )
    parser.add_argument("--dataset", default="coco_stuff")
    parser.add_argument("--data-root")
    parser.add_argument("--model", default="deeplabv3plus_r50_d8")
    parser.add_argument("--subset-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-name", default="result.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--samples-per-gpu", type=int, default=2)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--val-workers-per-gpu", type=int, default=2)
    parser.add_argument("--launcher", default="none", choices=["none", "pytorch", "slurm", "mpi"])
    parser.add_argument("--dist-backend", default="nccl")
    parser.add_argument("--gpu-collect", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    return parser


def _progress(message: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[supervised_probe] {message}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    _progress(
        "starting "
        f"dataset={args.dataset} model={args.model} seed={args.seed} max_iters={args.max_iters} "
        f"crop_size={args.crop_size} "
        f"subset_manifest={os.path.abspath(args.subset_manifest) if args.subset_manifest else '<full-train-split>'}"
    )
    result = run_supervised_probe(
        dataset_key=str(args.dataset),
        data_root=os.path.abspath(args.data_root) if args.data_root else None,
        model_key=str(args.model),
        subset_manifest_path=os.path.abspath(args.subset_manifest) if args.subset_manifest else None,
        output_dir=os.path.abspath(args.output_dir),
        result_name=str(args.result_name),
        seed=int(args.seed),
        max_iters=int(args.max_iters),
        crop_size=int(args.crop_size),
        samples_per_gpu=int(args.samples_per_gpu),
        workers_per_gpu=int(args.workers_per_gpu),
        val_workers_per_gpu=int(args.val_workers_per_gpu),
        launcher=str(args.launcher),
        dist_backend=str(args.dist_backend),
        gpu_collect=bool(args.gpu_collect),
    )
    if result is None:
        return 0
    summary = ((result.get("metrics") or {}).get("summary") or {})
    _progress(
        "finished "
        f"dataset={args.dataset} "
        f"mIoU={summary.get('mIoU')} "
        f"mAcc={summary.get('mAcc')} aAcc={summary.get('aAcc')}"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
