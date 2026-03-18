from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

import numpy as np

if __package__ in {None, ""}:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from slice_discovery.projector import SliceFeatureProjector
    from slice_remix.actions import generate_pairwise_candidates, select_pairwise_directions
    from slice_remix.baseline import estimate_baseline_mixture, load_slice_artifacts
    from slice_remix.dataset import read_jsonl
    from slice_remix.policy import compute_importance_weights, summarize_target_quotas
    from slice_remix.portraits import compute_portrait_shift, compute_slice_portraits, load_portrait_feature_groups
    from slice_remix.recommender import build_recommendation_result, rank_candidates
    from slice_remix.surrogate import BootstrapRemixSurrogate, LinearRemixSurrogate, QuadraticRemixSurrogate
else:
    from .slice_discovery.projector import SliceFeatureProjector
    from .slice_remix.actions import generate_pairwise_candidates, select_pairwise_directions
    from .slice_remix.baseline import estimate_baseline_mixture, load_slice_artifacts
    from .slice_remix.dataset import read_jsonl
    from .slice_remix.policy import compute_importance_weights, summarize_target_quotas
    from .slice_remix.portraits import compute_portrait_shift, compute_slice_portraits, load_portrait_feature_groups
    from .slice_remix.recommender import build_recommendation_result, rank_candidates
    from .slice_remix.surrogate import BootstrapRemixSurrogate, LinearRemixSurrogate, QuadraticRemixSurrogate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a remix surrogate and rank runtime recommendation candidates.")
    parser.add_argument("--projected-dir", required=True)
    parser.add_argument("--cluster-dir", required=True)
    parser.add_argument("--response-dataset", required=True)
    parser.add_argument("--baseline-seed", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output-path", default="./artifacts/remix_recommendation.json")
    parser.add_argument("--amplitudes", default="0.05,0.1")
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-support", type=float, default=0.0)
    parser.add_argument("--portrait-source", choices=["auto", "projected", "semantic"], default="auto")
    parser.add_argument("--processed-data-root")
    parser.add_argument("--schema-path")
    parser.add_argument("--assembled-feature-dir")
    parser.add_argument("--surrogate-model", choices=["linear", "quadratic"], default="linear")
    parser.add_argument("--bootstrap-models", type=int, default=1)
    parser.add_argument("--pair-selector", choices=["first", "portrait_diversity"], default="portrait_diversity")
    return parser


def _progress(message: str) -> None:
    print(f"[remix_recommendation] {message}", file=sys.stderr, flush=True)


def _parse_float_csv(raw: str) -> list[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def _ordered_pairs(num_slices: int, max_pairs: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for receiver in range(num_slices):
        for donor in range(num_slices):
            if receiver == donor:
                continue
            pairs.append((receiver, donor))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _select_pair_library(
    *,
    baseline_mixture: np.ndarray,
    portraits: dict[str, np.ndarray],
    max_pairs: int,
    amplitudes: list[float],
    pair_selector: str,
) -> list[tuple[int, int]]:
    ordered_pairs = _ordered_pairs(len(baseline_mixture), len(baseline_mixture) * (len(baseline_mixture) - 1))
    if pair_selector == "first":
        return ordered_pairs[:max_pairs]
    if pair_selector == "portrait_diversity":
        return select_pairwise_directions(
            baseline_mixture=baseline_mixture,
            portraits=portraits,
            max_pairs=max_pairs,
            ordered_pairs=ordered_pairs,
            min_amplitude=min(amplitudes) if amplitudes else 0.0,
        )
    raise ValueError(f"Unsupported pair_selector='{pair_selector}'")


def _build_surrogate(model_name: str, bootstrap_models: int):
    if bootstrap_models > 1:
        return BootstrapRemixSurrogate(base_model=model_name, num_models=bootstrap_models)
    if model_name == "linear":
        return LinearRemixSurrogate()
    if model_name == "quadratic":
        return QuadraticRemixSurrogate()
    raise ValueError(f"Unsupported surrogate_model='{model_name}'")


def run(args: argparse.Namespace, log_fn=_progress) -> int:
    log_fn("loading labeled response dataset")
    response_rows = [row for row in read_jsonl(os.path.abspath(args.response_dataset)) if row.get("measured_gain") is not None]
    if not response_rows:
        raise ValueError("response dataset must contain rows with measured_gain")

    log_fn(f"fitting surrogate model={args.surrogate_model} bootstrap_models={int(args.bootstrap_models)} rows={len(response_rows)}")
    surrogate = _build_surrogate(args.surrogate_model, int(args.bootstrap_models)).fit(response_rows)
    log_fn("loading projected artifacts")
    projected = SliceFeatureProjector.load(os.path.abspath(args.projected_dir))
    log_fn("loading slice artifacts")
    artifacts = load_slice_artifacts(os.path.abspath(args.cluster_dir))
    if projected.sample_ids != artifacts.sample_ids:
        raise ValueError("projected sample ids must match cluster sample ids")

    log_fn(f"loading portrait feature groups source={args.portrait_source}")
    feature_groups, portrait_source = load_portrait_feature_groups(
        projected=projected,
        cluster_meta=artifacts.meta,
        portrait_source=args.portrait_source,
        processed_data_root=os.path.abspath(args.processed_data_root) if args.processed_data_root else None,
        schema_path=os.path.abspath(args.schema_path) if args.schema_path else None,
        assembled_feature_dir=os.path.abspath(args.assembled_feature_dir) if args.assembled_feature_dir else None,
        log_fn=log_fn,
    )
    log_fn(f"computing slice portraits source={portrait_source}")
    portraits = compute_slice_portraits(feature_groups, artifacts.membership)
    rng = np.random.default_rng(int(args.baseline_seed))
    sample_indices = rng.choice(len(artifacts.sample_ids), size=int(args.budget), replace=False)
    baseline_mixture = estimate_baseline_mixture(artifacts.membership, sample_indices.tolist())
    amplitudes = _parse_float_csv(args.amplitudes)
    pair_library = _select_pair_library(
        baseline_mixture=baseline_mixture,
        portraits=portraits,
        max_pairs=int(args.max_pairs),
        amplitudes=amplitudes,
        pair_selector=args.pair_selector,
    )
    candidates = generate_pairwise_candidates(
        baseline_mixture,
        amplitudes=amplitudes,
        ordered_pairs=pair_library,
    )
    log_fn(f"generated runtime candidates count={len(candidates)} selector={args.pair_selector}")

    candidate_rows: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(candidates):
        target_mixture = np.asarray(candidate.target_mixture, dtype=np.float32)
        weights = compute_importance_weights(artifacts.membership, target_mixture)
        candidate_rows.append(
            {
                "candidate_id": f"cand_{args.baseline_seed}_{candidate_index}",
                "baseline_mixture": baseline_mixture.tolist(),
                "target_mixture": list(candidate.target_mixture),
                "delta_q": list(candidate.delta_q),
                "delta_phi": compute_portrait_shift(portraits, baseline_mixture, target_mixture),
                "context": {
                    "budget": int(args.budget),
                    "baseline_seed": int(args.baseline_seed),
                },
                "portrait_source": portrait_source,
                "support_size": int(candidate.support_size),
                "l1_shift": float(np.abs(np.asarray(candidate.delta_q, dtype=np.float32)).sum()),
                "rationale": {
                    "donors": list(candidate.donors),
                    "receivers": list(candidate.receivers),
                    "amplitude": float(candidate.amplitude),
                    "pair_selector": args.pair_selector,
                },
                "execution": {
                    "expected_slice_quotas": summarize_target_quotas(target_mixture, int(args.budget)),
                    "max_weight_sample_id": artifacts.sample_ids[int(np.argmax(weights))],
                },
            }
        )

    ranked = rank_candidates(
        candidate_rows,
        surrogate,
        kappa=float(args.kappa),
        lambda_l1=float(args.lambda_l1),
        lambda_support=float(args.lambda_support),
    )
    result = build_recommendation_result(ranked[0])

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log_fn(f"writing recommendation output={output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
