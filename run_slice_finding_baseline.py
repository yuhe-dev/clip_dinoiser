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
    from .slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from .slice_discovery.assembler import ProcessedFeatureAssembler
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
    parser = argparse.ArgumentParser(description="Run the slice discovery baseline pipeline on processed feature bundles.")
    parser.add_argument("--data-root", default="./data/data_feature")
    parser.add_argument("--schema-path", default="./docs/feature_schema/unified_processed_feature_schema.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--finder", choices=["soft_kmeans", "gmm", "vmf"], default="soft_kmeans")
    parser.add_argument("--num-slices", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--scalar-scaler", choices=["zscore", "none"], default="zscore")
    parser.add_argument("--block-weighting", choices=["equal_by_block", "none"], default="equal_by_block")
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


def _progress(message: str) -> None:
    print(f"[slice_baseline] {message}", file=sys.stderr, flush=True)


def _make_progress_callback(finder_name: str):
    def _callback(event: dict[str, object]) -> None:
        iteration = int(event["iteration"])
        max_iters = int(event["max_iters"])
        if finder_name == "gmm":
            log_likelihood = float(event["log_likelihood"])
            _progress(f"gmm iter {iteration}/{max_iters} log_likelihood={log_likelihood:.6f}")
        elif finder_name == "vmf":
            log_likelihood = float(event["log_likelihood"])
            mean_kappa = float(event["mean_kappa"])
            _progress(f"vmf iter {iteration}/{max_iters} log_likelihood={log_likelihood:.6f} mean_kappa={mean_kappa:.4f}")
        else:
            max_center_delta = float(event["max_center_delta"])
            _progress(f"soft_kmeans iter {iteration}/{max_iters} max_center_delta={max_center_delta:.6f}")

    return _callback


def _parse_candidate_ks(raw: str | None, *, min_num_slices: int, max_num_slices: int, sample_count: int) -> list[int]:
    if raw:
        candidate_ks = sorted({int(token.strip()) for token in raw.split(",") if token.strip()})
    else:
        candidate_ks = generate_candidate_ks(min_num_slices=min_num_slices, max_num_slices=max_num_slices)
    filtered = [value for value in candidate_ks if 1 <= value <= int(sample_count)]
    if not filtered:
        raise ValueError("candidate_ks must contain at least one value that does not exceed sample_count")
    return filtered


def run(args: argparse.Namespace) -> int:
    data_root = os.path.abspath(args.data_root)
    schema_path = os.path.abspath(args.schema_path)
    output_dir = os.path.abspath(args.output_dir)

    _progress("loading processed bundles")
    assembler = ProcessedFeatureAssembler.from_processed_paths(
        quality_path=os.path.join(data_root, "quality", "quality_processed_features.npy"),
        difficulty_path=os.path.join(data_root, "difficulty", "difficulty_processed_features.npy"),
        coverage_path=os.path.join(data_root, "coverage", "coverage_processed_features.npy"),
        schema_path=schema_path,
    )
    _progress("assembling sample-level features")
    projector = SliceFeatureProjector(
        scalar_scaler=args.scalar_scaler,
        block_weighting=args.block_weighting,
    )
    _progress("projecting features")
    projected = projector.fit_transform(assembler)
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
        candidate_ks = _parse_candidate_ks(
            args.candidate_ks,
            min_num_slices=int(args.min_num_slices),
            max_num_slices=int(args.max_num_slices),
            sample_count=int(projected.matrix.shape[0]),
        )
        candidate_rows = []
        for num_slices in candidate_ks:
            _progress(f"evaluating auto candidate k={num_slices}")
            if args.finder == "vmf":
                candidate = evaluate_vmf_candidate(
                    matrix=projected.matrix,
                    sample_ids=assembler.sample_ids,
                    num_slices=int(num_slices),
                    thresholds=thresholds,
                    seed=int(args.seed),
                    max_iters=int(args.max_iters),
                )
            else:
                candidate = evaluate_gmm_candidate(
                    matrix=projected.matrix,
                    sample_ids=assembler.sample_ids,
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
    result = finder.fit(
        projected.matrix,
        assembler.sample_ids,
        progress_callback=_make_progress_callback(args.finder),
    )

    _progress("writing artifacts")
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
        "sample_count": int(assembler.sample_count),
        "schema_path": schema_path,
        "data_root": data_root,
        "selection_mode": "auto" if args.auto_num_slices else "fixed",
        "selected_num_slices": int(selected_num_slices),
        "projector": {
            "scalar_scaler": args.scalar_scaler,
            "block_weighting": args.block_weighting,
        },
        "block_ranges": {
            name: [start, end] for name, (start, end) in projected.block_ranges.items()
        },
    }
    if selection_payload is not None:
        meta["candidate_ks"] = list(selection_payload["candidate_ks"])
    with open(os.path.join(output_dir, "slice_result_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    if selection_payload is not None:
        with open(os.path.join(output_dir, "model_selection.json"), "w", encoding="utf-8") as f:
            json.dump(selection_payload, f, indent=2, ensure_ascii=False)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
