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
    from slice_discovery.finder import GMMSliceFinder, SoftKMeansSliceFinder
    from slice_discovery.projector import SliceFeatureProjector
else:
    from .slice_discovery.finder import GMMSliceFinder, SoftKMeansSliceFinder
    from .slice_discovery.projector import SliceFeatureProjector


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run slice clustering on projected features and persist cluster-stage debug artifacts.")
    parser.add_argument("--projected-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--finder", choices=["soft_kmeans", "gmm"], default="soft_kmeans")
    parser.add_argument("--num-slices", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
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
    projected_dir = os.path.abspath(args.projected_dir)
    output_dir = os.path.abspath(args.output_dir)

    projected = SliceFeatureProjector.load(projected_dir)
    finder = _build_finder(args)
    result = finder.fit(projected.matrix, projected.sample_ids)

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
        "sample_count": len(projected.sample_ids),
        "projected_dir": projected_dir,
        "block_ranges": {name: [start, end] for name, (start, end) in projected.block_ranges.items()},
    }
    with open(os.path.join(output_dir, "slice_result_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    diagnostics = dict(result.diagnostics or {})
    diagnostics.update(
        {
            "finder": args.finder,
            "num_slices": int(args.num_slices),
            "input_matrix_shape": list(projected.matrix.shape),
            "input_all_finite": bool(np.isfinite(projected.matrix).all()),
            "hard_assignment_counts": np.bincount(result.hard_assignment, minlength=args.num_slices).astype(int).tolist(),
            "slice_weights": result.slice_weights.astype(float).tolist(),
        }
    )
    with open(os.path.join(output_dir, "cluster_debug.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)

    print(f"finder={args.finder}")
    print(f"membership_shape={tuple(result.membership.shape)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
