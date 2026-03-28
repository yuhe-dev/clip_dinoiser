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
    from slice_discovery.finder import GMMSliceFinder, SoftKMeansSliceFinder, VMFSliceFinder
    from slice_discovery.projector import SliceFeatureProjector
    from slice_discovery.selection import (
        SliceSelectionThresholds,
        evaluate_gmm_candidate,
        evaluate_vmf_candidate,
        generate_candidate_ks,
        select_best_candidate,
    )
else:
    from .slice_discovery.finder import GMMSliceFinder, SoftKMeansSliceFinder, VMFSliceFinder
    from .slice_discovery.projector import SliceFeatureProjector
    from .slice_discovery.selection import (
        SliceSelectionThresholds,
        evaluate_gmm_candidate,
        evaluate_vmf_candidate,
        generate_candidate_ks,
        select_best_candidate,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run slice clustering on projected features and persist cluster-stage debug artifacts.")
    parser.add_argument("--projected-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--finder", choices=["soft_kmeans", "gmm", "vmf"], default="soft_kmeans")
    parser.add_argument("--num-slices", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--auto-num-slices", action="store_true")
    parser.add_argument("--candidate-ks")
    parser.add_argument("--min-num-slices", type=int, default=4)
    parser.add_argument("--max-num-slices", type=int, default=64)
    parser.add_argument("--min-slice-weight", type=float, default=0.005)
    parser.add_argument("--min-hard-count", type=int, default=250)
    parser.add_argument("--min-avg-max-membership", type=float, default=0.90)
    parser.add_argument("--max-avg-entropy", type=float, default=0.20)
    parser.add_argument("--min-coherence", type=float, default=0.47)
    parser.add_argument("--bic-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--coherence-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--log-likelihood-relative-tolerance", type=float, default=0.05)
    return parser


def _progress(message: str) -> None:
    print(f"[slice_cluster_debug] {message}", file=sys.stderr, flush=True)


def _build_finder(args: argparse.Namespace):
    if args.finder == "soft_kmeans":
        return SoftKMeansSliceFinder(
            num_slices=args.num_slices,
            seed=args.seed,
            max_iters=args.max_iters,
            temperature=args.temperature,
        )
    if args.finder == "vmf":
        return VMFSliceFinder(
            num_slices=args.num_slices,
            seed=args.seed,
            max_iters=args.max_iters,
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

    _progress("loading projected artifacts")
    projected = SliceFeatureProjector.load(projected_dir)
    selected_num_slices = int(args.num_slices)
    selection_payload: dict[str, object] | None = None
    if args.auto_num_slices:
        if args.finder not in {"gmm", "vmf"}:
            raise ValueError("auto-num-slices is currently supported only for finder=gmm or finder=vmf")
        _progress("selecting num_slices automatically")
        thresholds = SliceSelectionThresholds(
            min_slice_weight=float(args.min_slice_weight),
            min_hard_count=int(args.min_hard_count),
            min_avg_max_membership=float(args.min_avg_max_membership),
            max_avg_entropy=float(args.max_avg_entropy),
            min_coherence=float(args.min_coherence),
            bic_relative_tolerance=float(args.bic_relative_tolerance),
            coherence_relative_tolerance=float(args.coherence_relative_tolerance),
            log_likelihood_relative_tolerance=float(args.log_likelihood_relative_tolerance),
        )
        if args.candidate_ks:
            candidate_ks = sorted({int(token.strip()) for token in args.candidate_ks.split(",") if token.strip()})
        else:
            candidate_ks = generate_candidate_ks(int(args.min_num_slices), int(args.max_num_slices))
        candidate_ks = [value for value in candidate_ks if value <= int(projected.matrix.shape[0])]
        candidate_rows = []
        for num_slices in candidate_ks:
            _progress(f"evaluating auto candidate k={num_slices}")
            if args.finder == "vmf":
                candidate = evaluate_vmf_candidate(
                    matrix=projected.matrix,
                    sample_ids=projected.sample_ids,
                    num_slices=int(num_slices),
                    thresholds=thresholds,
                    seed=int(args.seed),
                    max_iters=int(args.max_iters),
                )
            else:
                candidate = evaluate_gmm_candidate(
                    matrix=projected.matrix,
                    sample_ids=projected.sample_ids,
                    num_slices=int(num_slices),
                    thresholds=thresholds,
                    seed=int(args.seed),
                    max_iters=int(args.max_iters),
                )
            candidate_rows.append(candidate)
            _progress(
                "auto candidate "
                f"k={candidate.num_slices} admissible={candidate.admissible} "
                f"log_likelihood={candidate.log_likelihood:.4f} bic={candidate.bic:.4f} min_weight={candidate.min_slice_weight:.4f} "
                f"min_hard_count={candidate.min_hard_count} min_coherence={candidate.min_coherence:.4f}"
            )
        selected = select_best_candidate(candidate_rows, thresholds, finder=args.finder)
        selected_num_slices = int(selected.num_slices)
        selection_payload = {
            "selection_mode": "auto",
            "finder": args.finder,
            "selected_k": selected_num_slices,
            "candidate_ks": candidate_ks,
            "thresholds": {
                "min_slice_weight": float(thresholds.min_slice_weight),
                "min_hard_count": int(thresholds.min_hard_count),
                "min_avg_max_membership": float(thresholds.min_avg_max_membership),
                "max_avg_entropy": float(thresholds.max_avg_entropy),
                "min_coherence": float(thresholds.min_coherence),
                "bic_relative_tolerance": float(thresholds.bic_relative_tolerance),
                "coherence_relative_tolerance": float(thresholds.coherence_relative_tolerance),
                "log_likelihood_relative_tolerance": float(thresholds.log_likelihood_relative_tolerance),
            },
            "candidates": [candidate.to_dict() for candidate in candidate_rows],
        }

    finder_args = argparse.Namespace(**vars(args))
    finder_args.num_slices = selected_num_slices
    finder = _build_finder(finder_args)
    _progress(f"clustering with {args.finder} num_slices={selected_num_slices}")
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
        "num_slices": int(selected_num_slices),
        "sample_count": len(projected.sample_ids),
        "projected_dir": projected_dir,
        "selection_mode": "auto" if args.auto_num_slices else "fixed",
        "selected_num_slices": int(selected_num_slices),
        "block_ranges": {name: [start, end] for name, (start, end) in projected.block_ranges.items()},
    }
    if selection_payload is not None:
        meta["candidate_ks"] = list(selection_payload["candidate_ks"])
    with open(os.path.join(output_dir, "slice_result_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    diagnostics = dict(result.diagnostics or {})
    diagnostics.update(
        {
            "finder": args.finder,
            "num_slices": int(selected_num_slices),
            "input_matrix_shape": list(projected.matrix.shape),
            "input_all_finite": bool(np.isfinite(projected.matrix).all()),
            "hard_assignment_counts": np.bincount(result.hard_assignment, minlength=selected_num_slices).astype(int).tolist(),
            "slice_weights": result.slice_weights.astype(float).tolist(),
        }
    )
    with open(os.path.join(output_dir, "cluster_debug.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)
    if selection_payload is not None:
        with open(os.path.join(output_dir, "model_selection.json"), "w", encoding="utf-8") as f:
            json.dump(selection_payload, f, indent=2, ensure_ascii=False)

    print(f"finder={args.finder}")
    print(f"membership_shape={tuple(result.membership.shape)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
