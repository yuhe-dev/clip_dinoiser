import argparse
import json
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from clip_dinoiser.slice_remix.voc_feature_prep import (
    DEFAULT_AXIS_NAMES,
    AVAILABLE_AXIS_NAMES,
    DEFAULT_VOC_ROOT,
    prepare_voc_train_aug_feature_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare full train_aug feature tables and controlled manifests for VOC train_aug experiments."
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_VOC_ROOT,
        help="Path to the VOC2012 root that contains train_aug.txt and SegmentationClassAug.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write feature_table.jsonl, summary.json, and manifests/ into.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=2000,
        help="Number of samples to place into each prepared training subset.",
    )
    parser.add_argument(
        "--anchor-seed",
        type=int,
        default=0,
        help="Seed used to build the fixed anchor subset.",
    )
    parser.add_argument(
        "--candidate-budget",
        type=int,
        default=None,
        help="Optional cap on the extreme candidate pool before selecting high/low subsets.",
    )
    parser.add_argument(
        "--small-object-tau-ratio",
        type=float,
        default=0.02,
        help="Connected-component area ratio threshold used by the small-object scorer.",
    )
    parser.add_argument(
        "--rare-class-clip-percentile",
        type=float,
        default=95.0,
        help="Percentile ceiling for rare_class_exposure_clipped log inverse class-frequency weights.",
    )
    parser.add_argument(
        "--crop-survival-crop-size",
        type=int,
        default=512,
        help="Crop size used by crop_survival_score simulation.",
    )
    parser.add_argument(
        "--crop-survival-resize-ratio-min",
        type=float,
        default=0.5,
        help="Minimum supervised-probe resize ratio for crop_survival_score simulation.",
    )
    parser.add_argument(
        "--crop-survival-resize-ratio-max",
        type=float,
        default=2.0,
        help="Maximum supervised-probe resize ratio for crop_survival_score simulation.",
    )
    parser.add_argument(
        "--crop-survival-simulations",
        type=int,
        default=24,
        help="Monte Carlo trials per mask for crop_survival_score.",
    )
    parser.add_argument(
        "--crop-survival-seed",
        type=int,
        default=None,
        help="Optional seed for crop_survival_score simulation. Defaults to anchor seed.",
    )
    parser.add_argument(
        "--feature-axis",
        action="append",
        choices=list(AVAILABLE_AXIS_NAMES),
        help=(
            "Feature axis to materialize. Repeat this flag to request multiple axes. "
            f"Defaults to the current screening default axes: {', '.join(DEFAULT_AXIS_NAMES)}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = prepare_voc_train_aug_feature_experiment(
        data_root=args.data_root,
        output_dir=args.output_dir,
        subset_size=args.subset_size,
        anchor_seed=args.anchor_seed,
        candidate_budget=args.candidate_budget,
        small_object_tau_ratio=args.small_object_tau_ratio,
        rare_class_clip_percentile=args.rare_class_clip_percentile,
        crop_survival_crop_size=args.crop_survival_crop_size,
        crop_survival_resize_ratio_min=args.crop_survival_resize_ratio_min,
        crop_survival_resize_ratio_max=args.crop_survival_resize_ratio_max,
        crop_survival_simulations=args.crop_survival_simulations,
        crop_survival_seed=args.crop_survival_seed,
        feature_axes=args.feature_axis,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
