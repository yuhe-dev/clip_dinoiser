from __future__ import annotations

import argparse
import faulthandler
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
    from slice_discovery.projector import SliceFeatureProjector
    from slice_remix.actions import generate_pairwise_candidates, select_pairwise_directions
    from slice_remix.baseline import estimate_baseline_mixture, load_slice_artifacts
    from slice_remix.dataset import build_response_row, write_jsonl
    from slice_remix.policy import compute_importance_weights, sample_budgeted_subset, summarize_target_quotas
    from slice_remix.portraits import compute_portrait_shift, compute_slice_portraits, load_portrait_feature_groups
else:
    from .slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from .slice_discovery.projector import SliceFeatureProjector
    from .slice_remix.actions import generate_pairwise_candidates, select_pairwise_directions
    from .slice_remix.baseline import estimate_baseline_mixture, load_slice_artifacts
    from .slice_remix.dataset import build_response_row, write_jsonl
    from .slice_remix.policy import compute_importance_weights, sample_budgeted_subset, summarize_target_quotas
    from .slice_remix.portraits import compute_portrait_shift, compute_slice_portraits, load_portrait_feature_groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare remix response dataset rows from slice artifacts.")
    parser.add_argument("--projected-dir", required=True)
    parser.add_argument("--cluster-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--baseline-seeds", default="0")
    parser.add_argument("--amplitudes", default="0.05,0.1")
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--subset-manifest-dir")
    parser.add_argument("--pool-image-root")
    parser.add_argument("--portrait-source", choices=["auto", "projected", "semantic"], default="auto")
    parser.add_argument("--processed-data-root")
    parser.add_argument("--schema-path")
    parser.add_argument("--assembled-feature-dir")
    parser.add_argument("--pair-selector", choices=["first", "portrait_diversity"], default="portrait_diversity")
    return parser


def _parse_int_csv(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def _parse_float_csv(raw: str) -> list[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def _progress(message: str) -> None:
    print(f"[remix_response_dataset] {message}", file=sys.stderr, flush=True)


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


def _write_subset_manifest(
    manifest_dir: str,
    candidate_id: str,
    sample_ids: list[str],
    pool_image_root: str | None,
) -> str:
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, f"{candidate_id}.json")
    payload = {
        "candidate_id": candidate_id,
        "sample_ids": list(sample_ids),
    }
    if pool_image_root is not None:
        payload["sample_paths"] = [os.path.join(pool_image_root, sample_id) for sample_id in sample_ids]
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return manifest_path


def run(args: argparse.Namespace, log_fn=_progress) -> int:
    faulthandler.enable()
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
    amplitudes = _parse_float_csv(args.amplitudes)
    baseline_seeds = _parse_int_csv(args.baseline_seeds)
    log_fn(f"preparing baseline trials count={len(baseline_seeds)} amplitudes={amplitudes}")

    rows: list[dict[str, object]] = []
    for baseline_seed in baseline_seeds:
        log_fn(f"baseline_seed={baseline_seed} sampling baseline subset budget={int(args.budget)}")
        rng = np.random.default_rng(int(baseline_seed))
        sample_indices = rng.choice(len(artifacts.sample_ids), size=int(args.budget), replace=False)
        baseline_sample_ids = [artifacts.sample_ids[int(index)] for index in sample_indices.tolist()]
        baseline_mixture = estimate_baseline_mixture(artifacts.membership, sample_indices.tolist())
        baseline_manifest_path = None
        if args.subset_manifest_dir:
            baseline_manifest_path = _write_subset_manifest(
                os.path.abspath(args.subset_manifest_dir),
                f"baseline_{baseline_seed}",
                baseline_sample_ids,
                os.path.abspath(args.pool_image_root) if args.pool_image_root else None,
            )
            log_fn(f"baseline_seed={baseline_seed} wrote baseline manifest")
        pair_library = _select_pair_library(
            baseline_mixture=baseline_mixture,
            portraits=portraits,
            max_pairs=int(args.max_pairs),
            amplitudes=amplitudes,
            pair_selector=args.pair_selector,
        )
        log_fn(f"baseline_seed={baseline_seed} selected {len(pair_library)} pair directions selector={args.pair_selector}")
        candidates = generate_pairwise_candidates(
            baseline_mixture,
            amplitudes=amplitudes,
            ordered_pairs=pair_library,
        )
        log_fn(f"baseline_seed={baseline_seed} generated {len(candidates)} pairwise candidates")

        for candidate_index, candidate in enumerate(candidates):
            target_mixture = np.asarray(candidate.target_mixture, dtype=np.float32)
            candidate_id = f"cand_{baseline_seed}_{candidate_index}"
            row = build_response_row(
                baseline_trial_id=f"baseline_{baseline_seed}",
                candidate_id=candidate_id,
                baseline_mixture=baseline_mixture,
                target_mixture=target_mixture,
                delta_q=candidate.delta_q,
                delta_phi=compute_portrait_shift(portraits, baseline_mixture, target_mixture),
                context={
                    "budget": int(args.budget),
                    "baseline_seed": int(baseline_seed),
                },
                measured_gain=None,
            )
            row["portrait_source"] = portrait_source
            row["support_size"] = int(candidate.support_size)
            row["l1_shift"] = float(np.abs(np.asarray(candidate.delta_q, dtype=np.float32)).sum())
            row["rationale"] = {
                "donors": list(candidate.donors),
                "receivers": list(candidate.receivers),
                "amplitude": float(candidate.amplitude),
                "pair_selector": args.pair_selector,
            }
            weights = compute_importance_weights(artifacts.membership, target_mixture)
            selected_ids = sample_budgeted_subset(
                artifacts.sample_ids,
                weights,
                budget=int(args.budget),
                seed=int(baseline_seed) + candidate_index,
            )
            row["execution"] = {
                "expected_slice_quotas": summarize_target_quotas(target_mixture, int(args.budget)),
                "max_weight_sample_id": artifacts.sample_ids[int(np.argmax(weights))],
                "selected_sample_ids": selected_ids,
            }
            if baseline_manifest_path is not None:
                row["execution"]["baseline_manifest_path"] = baseline_manifest_path
            if args.subset_manifest_dir:
                row["execution"]["subset_manifest_path"] = _write_subset_manifest(
                    os.path.abspath(args.subset_manifest_dir),
                    candidate_id,
                    selected_ids,
                os.path.abspath(args.pool_image_root) if args.pool_image_root else None,
            )
            rows.append(row)

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log_fn(f"writing response rows count={len(rows)} output={output_path}")
    write_jsonl(output_path, rows)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
