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
    from slice_remix.beam_search import (
        BeamSearchConfig,
        SearchEdge,
        TargetBeamSearchConfig,
        generate_beam_candidates,
        generate_target_beam_candidates,
    )
    from slice_remix.actions import generate_pairwise_candidates, select_pairwise_directions
    from slice_remix.baseline import estimate_baseline_mixture, load_slice_artifacts
    from slice_remix.class_coverage import load_class_presence_matrix, select_focus_class_spec
    from slice_remix.dataset import build_response_row, write_jsonl
    from slice_remix.policy import compute_importance_weights, materialize_budgeted_subset, summarize_target_quotas
    from slice_remix.portraits import build_feature_label_map, compute_portrait_shift, compute_slice_portraits, load_portrait_feature_groups
    from slice_remix.prior_graph import (
        SearchBias,
        SearchConstraints,
        TargetPortraitSpec,
        build_pool_target_portrait_spec,
        build_portrait_residual_context,
        build_prior_graph,
        build_target_prior_graph,
        build_target_residual_context,
    )
    from slice_remix.realized_features import (
        aggregate_feature_groups_by_indices,
        build_sample_index,
        resolve_sample_indices,
        serialize_feature_groups,
        subtract_feature_groups,
        summarize_feature_groups,
    )
else:
    from .slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from .slice_discovery.projector import SliceFeatureProjector
    from .slice_remix.beam_search import (
        BeamSearchConfig,
        SearchEdge,
        TargetBeamSearchConfig,
        generate_beam_candidates,
        generate_target_beam_candidates,
    )
    from .slice_remix.actions import generate_pairwise_candidates, select_pairwise_directions
    from .slice_remix.baseline import estimate_baseline_mixture, load_slice_artifacts
    from .slice_remix.class_coverage import load_class_presence_matrix, select_focus_class_spec
    from .slice_remix.dataset import build_response_row, write_jsonl
    from .slice_remix.policy import compute_importance_weights, materialize_budgeted_subset, summarize_target_quotas
    from .slice_remix.portraits import build_feature_label_map, compute_portrait_shift, compute_slice_portraits, load_portrait_feature_groups
    from .slice_remix.prior_graph import (
        SearchBias,
        SearchConstraints,
        TargetPortraitSpec,
        build_pool_target_portrait_spec,
        build_portrait_residual_context,
        build_prior_graph,
        build_target_prior_graph,
        build_target_residual_context,
    )
    from .slice_remix.realized_features import (
        aggregate_feature_groups_by_indices,
        build_sample_index,
        resolve_sample_indices,
        serialize_feature_groups,
        subtract_feature_groups,
        summarize_feature_groups,
    )


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
    parser.add_argument("--pair-selector", choices=["first", "portrait_diversity", "beam_v1", "beam_target_v1"], default="portrait_diversity")
    parser.add_argument("--annotation-root")
    parser.add_argument("--baseline-result-path")
    parser.add_argument("--full-result-path")
    parser.add_argument("--focus-class-gap-threshold", type=float, default=10.0)
    parser.add_argument("--focus-class-top-k", type=int, default=25)
    parser.add_argument("--coverage-alpha", type=float, default=0.25)
    parser.add_argument("--coverage-repair-budget", type=int, default=64)
    return parser


def _parse_int_csv(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def _parse_float_csv(raw: str) -> list[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def _progress(message: str) -> None:
    print(f"[remix_response_dataset] {message}", file=sys.stderr, flush=True)


def _load_optional_coverage_inputs(args: argparse.Namespace, sample_ids: list[str]) -> tuple[np.ndarray | None, dict[str, object] | None]:
    if not (args.annotation_root and args.baseline_result_path and args.full_result_path):
        return None, None
    with open(os.path.abspath(args.baseline_result_path), "r", encoding="utf-8") as f:
        baseline_result = json.load(f)
    with open(os.path.abspath(args.full_result_path), "r", encoding="utf-8") as f:
        full_result = json.load(f)
    focus_spec = select_focus_class_spec(
        baseline_result=baseline_result,
        full_result=full_result,
        min_iou_gap=float(args.focus_class_gap_threshold),
        top_k=int(args.focus_class_top_k),
    )
    if not focus_spec["class_indices"]:
        return None, focus_spec
    class_names = list(((full_result.get("coco_stuff") or {}).get("per_class") or {}).keys())
    class_presence = load_class_presence_matrix(
        sample_ids=sample_ids,
        annotation_root=os.path.abspath(args.annotation_root),
        num_classes=len(class_names),
    )
    return class_presence, focus_spec


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


def _slice_ids(num_slices: int) -> list[str]:
    return [f"slice_{index:02d}" for index in range(num_slices)]


def _payload_to_search_edges(payload: object, slice_ids: list[str]) -> list[SearchEdge]:
    index_by_id = {slice_id: index for index, slice_id in enumerate(slice_ids)}
    edges: list[SearchEdge] = []
    for edge in payload.edges:
        if not edge.admissible:
            continue
        edges.append(
            SearchEdge(
                donor=index_by_id[edge.donor],
                receiver=index_by_id[edge.receiver],
                score=float(edge.score),
                amplitude_band=tuple(float(value) for value in edge.amplitude_band),
                balance_score=float(edge.balance_score),
                risk_score=float(edge.risk_score),
                fit_score=float(getattr(edge, "fit_score", edge.score)),
                bias_score=float(getattr(edge, "bias_score", 0.0)),
            )
        )
    return edges


def _shift_quality_laplacian_target(target_spec: TargetPortraitSpec, *, block_name: str = "quality.laplacian", mass: float = 0.08) -> TargetPortraitSpec:
    if block_name not in target_spec.shape_targets:
        return target_spec
    original = np.asarray(target_spec.shape_targets[block_name], dtype=np.float32)
    if original.ndim != 1 or original.size < 3:
        return target_spec

    updated = original.copy()
    low_count = max(1, min(2, updated.size // 3 if updated.size >= 6 else 1))
    source_indices = np.arange(low_count, dtype=np.int64)
    target_start = max(low_count + 1, updated.size // 2)
    target_indices = np.arange(target_start, updated.size, dtype=np.int64)
    if target_indices.size == 0:
        target_indices = np.arange(updated.size - 1, updated.size, dtype=np.int64)

    removable = float(updated[source_indices].sum())
    transfer = min(float(mass), max(0.0, removable * 0.5))
    if transfer <= 1e-8:
        return target_spec

    source_weights = updated[source_indices]
    source_total = float(source_weights.sum())
    if source_total > 1e-8:
        updated[source_indices] -= transfer * (source_weights / source_total)
    target_weights = np.linspace(1.0, 2.0, num=target_indices.size, dtype=np.float32)
    target_weights /= float(target_weights.sum())
    updated[target_indices] += transfer * target_weights
    updated = np.clip(updated, 0.0, None)
    total = float(updated.sum())
    if total > 1e-8:
        updated /= total
    return TargetPortraitSpec(
        shape_targets={
            name: (updated.astype(np.float32) if name == block_name else np.asarray(values, dtype=np.float32).copy())
            for name, values in target_spec.shape_targets.items()
        },
        scalar_targets={
            name: np.asarray(values, dtype=np.float32).copy()
            for name, values in target_spec.scalar_targets.items()
        },
        block_weights={name: float(weight) for name, weight in target_spec.block_weights.items()},
        source="quality_laplacian_smooth_shift",
    )


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
    feature_label_map = build_feature_label_map(
        feature_groups,
        schema_path=os.path.abspath(args.schema_path) if args.schema_path else None,
    )
    sample_index = build_sample_index(artifacts.sample_ids)
    amplitudes = _parse_float_csv(args.amplitudes)
    baseline_seeds = _parse_int_csv(args.baseline_seeds)
    class_presence, focus_spec = _load_optional_coverage_inputs(args, artifacts.sample_ids)
    log_fn(f"preparing baseline trials count={len(baseline_seeds)} amplitudes={amplitudes}")

    rows: list[dict[str, object]] = []
    for baseline_seed in baseline_seeds:
        log_fn(f"baseline_seed={baseline_seed} sampling baseline subset budget={int(args.budget)}")
        rng = np.random.default_rng(int(baseline_seed))
        sample_indices = rng.choice(len(artifacts.sample_ids), size=int(args.budget), replace=False)
        baseline_sample_ids = [artifacts.sample_ids[int(index)] for index in sample_indices.tolist()]
        baseline_mixture = estimate_baseline_mixture(artifacts.membership, sample_indices.tolist())
        baseline_realized_features = aggregate_feature_groups_by_indices(
            feature_groups,
            sample_indices.tolist(),
        )
        baseline_manifest_path = None
        if args.subset_manifest_dir:
            baseline_manifest_path = _write_subset_manifest(
                os.path.abspath(args.subset_manifest_dir),
                f"baseline_{baseline_seed}",
                baseline_sample_ids,
                os.path.abspath(args.pool_image_root) if args.pool_image_root else None,
            )
            log_fn(f"baseline_seed={baseline_seed} wrote baseline manifest")
        if args.pair_selector == "beam_v1":
            slice_ids = _slice_ids(artifacts.membership.shape[1])
            portrait_context = build_portrait_residual_context(
                feature_groups=feature_groups,
                feature_label_map=feature_label_map,
                memberships=artifacts.membership,
                baseline_sample_indices=sample_indices.tolist(),
            )
            payload = build_prior_graph(
                feature_groups=feature_groups,
                feature_label_map=feature_label_map,
                memberships=artifacts.membership,
                baseline_sample_indices=sample_indices.tolist(),
                slice_ids=slice_ids,
                portrait_context=portrait_context,
                baseline_seed=int(baseline_seed),
                budget=int(args.budget),
            )
            candidates = generate_beam_candidates(
                baseline_mixture=baseline_mixture,
                pool_mixture=artifacts.membership.mean(axis=0, dtype=np.float32),
                edges=_payload_to_search_edges(payload, slice_ids),
                portrait_context=portrait_context,
                config=BeamSearchConfig(
                    max_depth=min(4, max(2, len(baseline_mixture) - 1)),
                    beam_width=max(8, int(args.max_pairs) * 2),
                    proposal_edges_per_node=max(12, int(args.max_pairs) * 4),
                ),
            )
            log_fn(f"baseline_seed={baseline_seed} generated {len(candidates)} beam candidates")
        elif args.pair_selector == "beam_target_v1":
            slice_ids = _slice_ids(artifacts.membership.shape[1])
            pool_target = build_pool_target_portrait_spec(
                feature_groups=feature_groups,
                feature_label_map=feature_label_map,
                memberships=artifacts.membership,
            )
            target_spec = _shift_quality_laplacian_target(pool_target)
            target_context = build_target_residual_context(
                feature_groups=feature_groups,
                feature_label_map=feature_label_map,
                memberships=artifacts.membership,
                baseline_sample_indices=sample_indices.tolist(),
                target_spec=target_spec,
            )
            payload = build_target_prior_graph(
                feature_groups=feature_groups,
                feature_label_map=feature_label_map,
                memberships=artifacts.membership,
                baseline_sample_indices=sample_indices.tolist(),
                slice_ids=slice_ids,
                target_spec=target_spec,
                target_context=target_context,
                constraints=SearchConstraints(),
                bias=SearchBias(),
                baseline_seed=int(baseline_seed),
                budget=int(args.budget),
            )
            candidates = generate_target_beam_candidates(
                baseline_mixture=baseline_mixture,
                edges=_payload_to_search_edges(payload, slice_ids),
                target_context=target_context,
                config=TargetBeamSearchConfig(
                    max_depth=min(4, max(2, len(baseline_mixture) - 1)),
                    beam_width=max(8, int(args.max_pairs) * 2),
                    proposal_edges_per_node=max(12, int(args.max_pairs) * 4),
                ),
            )
            log_fn(f"baseline_seed={baseline_seed} generated {len(candidates)} target-beam candidates")
        else:
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
            expected_delta_phi = compute_portrait_shift(portraits, baseline_mixture, target_mixture)
            weights = compute_importance_weights(artifacts.membership, target_mixture)
            materialization = materialize_budgeted_subset(
                artifacts.sample_ids,
                weights,
                budget=int(args.budget),
                seed=int(baseline_seed) + candidate_index,
                memberships=artifacts.membership,
                target_mixture=target_mixture,
                class_presence=class_presence,
                focus_class_indices=(focus_spec or {}).get("class_indices"),
                focus_class_weights=np.asarray((focus_spec or {}).get("class_weights", []), dtype=np.float32) if focus_spec else None,
                coverage_alpha=float(args.coverage_alpha),
                coverage_repair_budget=int(args.coverage_repair_budget),
            )
            selected_ids = list(materialization.selected_ids)
            selected_indices = list(materialization.selected_indices) or resolve_sample_indices(sample_index, selected_ids)
            target_realized_features = aggregate_feature_groups_by_indices(
                feature_groups,
                selected_indices,
            )
            realized_delta_phi = subtract_feature_groups(
                target_realized_features,
                baseline_realized_features,
            )
            row = build_response_row(
                baseline_trial_id=f"baseline_{baseline_seed}",
                candidate_id=candidate_id,
                baseline_mixture=baseline_mixture,
                target_mixture=target_mixture,
                delta_q=candidate.delta_q,
                delta_phi=realized_delta_phi,
                context={
                    "budget": int(args.budget),
                    "baseline_seed": int(baseline_seed),
                },
                measured_gain=None,
            )
            row["portrait_source"] = portrait_source
            row["feature_description_mode"] = "realized_sample_aggregation"
            row["baseline_features_raw"] = serialize_feature_groups(baseline_realized_features)
            row["target_features_raw"] = serialize_feature_groups(target_realized_features)
            row["baseline_features_summary"] = summarize_feature_groups(baseline_realized_features)
            row["target_features_summary"] = summarize_feature_groups(target_realized_features)
            row["expected_delta_phi"] = serialize_feature_groups(expected_delta_phi)
            row["support_size"] = int(candidate.support_size)
            row["l1_shift"] = float(np.abs(np.asarray(candidate.delta_q, dtype=np.float32)).sum())
            row["rationale"] = {
                "donors": list(candidate.donors),
                "receivers": list(candidate.receivers),
                "amplitude": float(candidate.amplitude),
                "pair_selector": args.pair_selector,
                "plan": list(candidate.metadata.get("plan", [])),
            }
            row["execution"] = {
                "expected_slice_quotas": summarize_target_quotas(target_mixture, int(args.budget)),
                "max_weight_sample_id": artifacts.sample_ids[int(np.argmax(weights))],
                "selected_sample_ids": selected_ids,
                "selection_seed": int(baseline_seed) + candidate_index,
                "materialization_policy": "quota_mixture_coverage_v1" if class_presence is not None else "quota_mixture_v2",
                "focus_class_indices": list((focus_spec or {}).get("class_indices", [])),
                "focus_class_names": list((focus_spec or {}).get("class_names", [])),
                "realized_mixture": [float(value) for value in np.asarray(materialization.realized_mixture, dtype=np.float32).tolist()],
                "mixture_l1_before_coverage_repair": (
                    None if materialization.mixture_l1_before_coverage_repair is None
                    else float(materialization.mixture_l1_before_coverage_repair)
                ),
                "mixture_l1_after_coverage_repair": float(materialization.mixture_l1_after_coverage_repair),
                "focus_coverage_before": [int(value) for value in materialization.focus_coverage_before],
                "focus_coverage_after": [int(value) for value in materialization.focus_coverage_after],
                "accepted_coverage_swaps": int(materialization.accepted_coverage_swaps),
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
