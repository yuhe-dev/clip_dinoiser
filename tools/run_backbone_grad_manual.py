#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
from pathlib import Path


VARIANT_MODULES = {
    "final": {
        "L0": [
            "obj_proj",
            "bkg_decoder",
        ],
        "L1": [
            "obj_proj",
            "bkg_decoder",
            "clip_backbone.decode_head.proj",
        ],
        "L2": [
            "obj_proj",
            "bkg_decoder",
            "clip_backbone.decode_head.proj",
            "clip_backbone.backbone.visual.transformer.resblocks.-1",
            "clip_backbone.backbone.visual.ln_post",
        ],
    },
    "intermediate": {
        "L0": [
            "obj_proj",
            "bkg_decoder",
        ],
        "L1": [
            "obj_proj",
            "bkg_decoder",
            "clip_backbone.backbone.visual.transformer.resblocks.-3",
        ],
        "L2": [
            "obj_proj",
            "bkg_decoder",
            "clip_backbone.backbone.visual.transformer.resblocks.-3",
            "clip_backbone.backbone.visual.transformer.resblocks.-4",
        ],
    },
}

DEFAULT_CONFIG_BY_FAMILY = {
    "final": "feature_experiment_fast_cached_slide_backbone_grad",
    "intermediate": "feature_experiment_fast_cached_slide_intermediate_grad",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a manual backbone-grad training command for CLIP-DINOiser learner audits."
    )
    parser.add_argument("--family", choices=sorted(VARIANT_MODULES), default="final")
    parser.add_argument("--variant", choices=("L0", "L1", "L2"), required=True)
    parser.add_argument("--subset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-name", default="result.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--master-port", type=int, default=29741)
    parser.add_argument(
        "--config",
        default=None,
        help="Hydra config name for the learner branch. Defaults to the selected family.",
    )
    parser.add_argument(
        "--python-bin",
        default="/home/yuhe/.conda/envs/clipdino2/bin/python3.9",
    )
    parser.add_argument(
        "--workdir",
        default="/home/yuhe/clip_dinoiser",
    )
    return parser


def build_command(args: argparse.Namespace) -> str:
    config_name = args.config or DEFAULT_CONFIG_BY_FAMILY[str(args.family)]
    lines = [
        f"cd {shlex.quote(args.workdir)}",
        "",
        "MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU "
        f"CUDA_VISIBLE_DEVICES={shlex.quote(str(args.gpu))} \\",
        f"{shlex.quote(args.python_bin)} -m torch.distributed.run \\",
        "  --nproc_per_node=1 \\",
        f"  --master_port {int(args.master_port)} \\",
        "  run_remix_training_experiment.py \\",
        f"  --config {shlex.quote(config_name)} \\",
        f"  --subset-manifest {shlex.quote(args.subset_manifest)} \\",
        f"  --output-dir {shlex.quote(args.output_dir)} \\",
        f"  --result-name {shlex.quote(args.result_name)} \\",
        f"  --seed {int(args.seed)} \\",
    ]
    modules = VARIANT_MODULES[str(args.family)][str(args.variant)]
    for index, module_path in enumerate(modules):
        suffix = " \\" if index < len(modules) - 1 else ""
        lines.append(f"  --trainable-modules {shlex.quote(module_path)}{suffix}")
    return "\n".join(lines)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    print(build_command(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
