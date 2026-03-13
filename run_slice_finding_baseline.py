from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

if __package__ in {None, ""}:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from slice_discovery.assembler import ProcessedFeatureAssembler
    from slice_discovery.finder import GMMSliceFinder, SoftKMeansSliceFinder
    from slice_discovery.projector import SliceFeatureProjector
else:
    from .slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from .slice_discovery.assembler import ProcessedFeatureAssembler
    from .slice_discovery.finder import GMMSliceFinder, SoftKMeansSliceFinder
    from .slice_discovery.projector import SliceFeatureProjector


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the slice discovery baseline pipeline on processed feature bundles.")
    parser.add_argument("--data-root", default="./data/data_feature")
    parser.add_argument("--schema-path", default="./docs/feature_schema/unified_processed_feature_schema.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--finder", choices=["soft_kmeans", "gmm"], default="soft_kmeans")
    parser.add_argument("--num-slices", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--scalar-scaler", choices=["zscore", "none"], default="zscore")
    parser.add_argument("--block-weighting", choices=["equal_by_block", "none"], default="equal_by_block")
    return parser


def _build_finder(args: argparse.Namespace):
    if args.finder == "soft_kmeans":
        return SoftKMeansSliceFinder(
            num_slices=args.num_slices,
            seed=args.seed,
            max_iters=args.max_iters,
            temperature=args.temperature,
        )
    return GMMSliceFinder(
        num_slices=args.num_slices,
        seed=args.seed,
        max_iters=args.max_iters,
        covariance_type="diag",
    )


def run(args: argparse.Namespace) -> int:
    data_root = os.path.abspath(args.data_root)
    schema_path = os.path.abspath(args.schema_path)
    output_dir = os.path.abspath(args.output_dir)

    assembler = ProcessedFeatureAssembler.from_processed_paths(
        quality_path=os.path.join(data_root, "quality", "quality_processed_features.npy"),
        difficulty_path=os.path.join(data_root, "difficulty", "difficulty_processed_features.npy"),
        coverage_path=os.path.join(data_root, "coverage", "coverage_processed_features.npy"),
        schema_path=schema_path,
    )
    projector = SliceFeatureProjector(
        scalar_scaler=args.scalar_scaler,
        block_weighting=args.block_weighting,
    )
    projected = projector.fit_transform(assembler)
    finder = _build_finder(args)
    result = finder.fit(projected.matrix, assembler.sample_ids)

    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, "slice_result.npz"),
        sample_ids=np.asarray(result.sample_ids, dtype=object),
        membership=result.membership,
        hard_assignment=result.hard_assignment,
        slice_weights=result.slice_weights,
        centers=result.centers,
    )

    meta = {
        "finder": args.finder,
        "num_slices": int(args.num_slices),
        "sample_count": int(assembler.sample_count),
        "schema_path": schema_path,
        "data_root": data_root,
        "projector": {
            "scalar_scaler": args.scalar_scaler,
            "block_weighting": args.block_weighting,
        },
        "block_ranges": {
            name: [start, end] for name, (start, end) in projected.block_ranges.items()
        },
    }
    with open(os.path.join(output_dir, "slice_result_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
