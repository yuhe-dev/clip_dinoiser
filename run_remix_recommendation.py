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
    from slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from slice_discovery.projector import SliceFeatureProjector
    from slice_remix.beam_search import (
        BeamSearchConfig,
        SearchEdge,
        TargetBeamSearchConfig,
        generate_beam_candidates_with_trace,
        generate_target_beam_candidates_with_trace,
    )
    from slice_remix.actions import generate_pairwise_candidates, select_pairwise_directions
    from slice_remix.baseline import estimate_baseline_mixture, load_slice_artifacts
    from slice_remix.dataset import read_jsonl
    from slice_remix.policy import compute_importance_weights, materialize_budgeted_subset, summarize_target_quotas
    from slice_remix.portraits import (
        build_feature_label_map,
        compute_portrait_shift,
        compute_slice_portraits,
        load_portrait_feature_groups,
        summarize_portrait_shift,
    )
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
    from slice_remix.recommender import build_recommendation_result, rank_candidates
    from slice_remix.realized_features import (
        aggregate_feature_groups_by_indices,
        build_sample_index,
        resolve_sample_indices,
        serialize_feature_groups,
        subtract_feature_groups,
        summarize_feature_groups,
    )
    from slice_remix.surrogate import build_surrogate
else:
    from .slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from .slice_discovery.projector import SliceFeatureProjector
    from .slice_remix.beam_search import (
        BeamSearchConfig,
        SearchEdge,
        TargetBeamSearchConfig,
        generate_beam_candidates_with_trace,
        generate_target_beam_candidates_with_trace,
    )
    from .slice_remix.actions import generate_pairwise_candidates, select_pairwise_directions
    from .slice_remix.baseline import estimate_baseline_mixture, load_slice_artifacts
    from .slice_remix.dataset import read_jsonl
    from .slice_remix.policy import compute_importance_weights, materialize_budgeted_subset, summarize_target_quotas
    from .slice_remix.portraits import (
        build_feature_label_map,
        compute_portrait_shift,
        compute_slice_portraits,
        load_portrait_feature_groups,
        summarize_portrait_shift,
    )
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
    from .slice_remix.recommender import build_recommendation_result, rank_candidates
    from .slice_remix.realized_features import (
        aggregate_feature_groups_by_indices,
        build_sample_index,
        resolve_sample_indices,
        serialize_feature_groups,
        subtract_feature_groups,
        summarize_feature_groups,
    )
    from .slice_remix.surrogate import build_surrogate


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
    parser.add_argument("--pair-selector", choices=["first", "portrait_diversity", "beam_v1", "beam_target_v1"], default="portrait_diversity")
    parser.add_argument("--surrogate-output-path")
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


def _default_surrogate_output_path(recommendation_output_path: str) -> str:
    base, _ = os.path.splitext(recommendation_output_path)
    return f"{base}_surrogate.json"


def _attach_search_tree_metadata(
    *,
    trace: dict[str, object],
    ranked: list[dict[str, object]],
    slice_ids: list[str],
) -> dict[str, object]:
    ranked_candidate_ids_by_node: dict[str, tuple[str, int]] = {}
    for rank_index, candidate in enumerate(ranked, start=1):
        search_node_id = str(candidate.get("search_node_id", ""))
        candidate_id = str(candidate.get("candidate_id", ""))
        if not search_node_id or not candidate_id:
            continue
        ranked_candidate_ids_by_node[search_node_id] = (candidate_id, rank_index)

    recommended_candidate_id = str(ranked[0].get("candidate_id", "")) if ranked else ""
    raw_nodes = [dict(node) for node in trace.get("nodes", [])]
    child_counts: dict[str, int] = {}
    for node in raw_nodes:
        parent_id = node.get("parent_id")
        if isinstance(parent_id, str):
            child_counts[parent_id] = child_counts.get(parent_id, 0) + 1

    nodes: list[dict[str, object]] = []
    for node in raw_nodes:
        enriched = dict(node)
        candidate_id, rank_index = ranked_candidate_ids_by_node.get(str(enriched.get("node_id", "")), ("", None))
        enriched["candidate_id"] = candidate_id or None
        enriched["candidate_rank"] = rank_index
        enriched["is_recommended"] = bool(candidate_id and candidate_id == recommended_candidate_id)
        action = enriched.get("action")
        if isinstance(action, dict):
            donor = action.get("donor")
            receiver = action.get("receiver")
            if isinstance(donor, int) and 0 <= donor < len(slice_ids):
                action["donor"] = slice_ids[donor]
            if isinstance(receiver, int) and 0 <= receiver < len(slice_ids):
                action["receiver"] = slice_ids[receiver]
        translated_plan: list[dict[str, object]] = []
        for step in enriched.get("plan", []) or []:
            translated_step = dict(step)
            donor = translated_step.get("donor")
            receiver = translated_step.get("receiver")
            if isinstance(donor, int) and 0 <= donor < len(slice_ids):
                translated_step["donor"] = slice_ids[donor]
            if isinstance(receiver, int) and 0 <= receiver < len(slice_ids):
                translated_step["receiver"] = slice_ids[receiver]
            translated_plan.append(translated_step)
        enriched["plan"] = translated_plan
        if enriched.get("node_type") != "root" and candidate_id and child_counts.get(str(enriched.get("node_id", "")), 0) == 0:
            enriched["node_type"] = "completed"
        nodes.append(enriched)

    return {
        "root_id": trace.get("root_id"),
        "nodes": nodes,
    }


def run(args: argparse.Namespace, log_fn=_progress) -> int:
    output_path = os.path.abspath(args.output_path)
    surrogate_output_path = os.path.abspath(args.surrogate_output_path) if args.surrogate_output_path else _default_surrogate_output_path(output_path)
    log_fn("loading labeled response dataset")
    response_rows = [row for row in read_jsonl(os.path.abspath(args.response_dataset)) if row.get("measured_gain") is not None]
    if not response_rows:
        raise ValueError("response dataset must contain rows with measured_gain")

    log_fn(f"fitting surrogate model={args.surrogate_model} bootstrap_models={int(args.bootstrap_models)} rows={len(response_rows)}")
    surrogate = build_surrogate(args.surrogate_model, bootstrap_models=int(args.bootstrap_models)).fit(response_rows)
    log_fn(f"writing surrogate artifact output={surrogate_output_path}")
    surrogate.save_json(surrogate_output_path)
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
    rng = np.random.default_rng(int(args.baseline_seed))
    sample_indices = rng.choice(len(artifacts.sample_ids), size=int(args.budget), replace=False)
    baseline_mixture = estimate_baseline_mixture(artifacts.membership, sample_indices.tolist())
    baseline_realized_features = aggregate_feature_groups_by_indices(
        feature_groups,
        sample_indices.tolist(),
    )
    amplitudes = _parse_float_csv(args.amplitudes)
    search_tree_trace: dict[str, object] | None = None
    search_tree_slice_ids: list[str] = []
    if args.pair_selector == "beam_v1":
        slice_ids = _slice_ids(artifacts.membership.shape[1])
        search_tree_slice_ids = slice_ids
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
            baseline_seed=int(args.baseline_seed),
            budget=int(args.budget),
        )
        candidates, search_tree_trace = generate_beam_candidates_with_trace(
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
    elif args.pair_selector == "beam_target_v1":
        slice_ids = _slice_ids(artifacts.membership.shape[1])
        search_tree_slice_ids = slice_ids
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
            baseline_seed=int(args.baseline_seed),
            budget=int(args.budget),
        )
        candidates, search_tree_trace = generate_target_beam_candidates_with_trace(
            baseline_mixture=baseline_mixture,
            edges=_payload_to_search_edges(payload, slice_ids),
            target_context=target_context,
            config=TargetBeamSearchConfig(
                max_depth=min(4, max(2, len(baseline_mixture) - 1)),
                beam_width=max(8, int(args.max_pairs) * 2),
                proposal_edges_per_node=max(12, int(args.max_pairs) * 4),
            ),
        )
    else:
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
        selection_seed = int(args.baseline_seed) + candidate_index
        materialization = materialize_budgeted_subset(
            artifacts.sample_ids,
            weights,
            budget=int(args.budget),
            seed=selection_seed,
            memberships=artifacts.membership,
            target_mixture=target_mixture,
        )
        selected_ids = list(materialization.selected_ids)
        selected_indices = list(materialization.selected_indices) or resolve_sample_indices(sample_index, selected_ids)
        target_realized_features = aggregate_feature_groups_by_indices(
            feature_groups,
            selected_indices,
        )
        delta_phi = subtract_feature_groups(target_realized_features, baseline_realized_features)
        expected_delta_phi = compute_portrait_shift(portraits, baseline_mixture, target_mixture)
        candidate_rows.append(
            {
                "candidate_id": f"cand_{args.baseline_seed}_{candidate_index}",
                "baseline_mixture": baseline_mixture.tolist(),
                "target_mixture": list(candidate.target_mixture),
                "delta_q": list(candidate.delta_q),
                "delta_phi": delta_phi,
                "feature_description_mode": "realized_sample_aggregation",
                "baseline_features_raw": serialize_feature_groups(baseline_realized_features),
                "target_features_raw": serialize_feature_groups(target_realized_features),
                "baseline_features_summary": summarize_feature_groups(baseline_realized_features),
                "target_features_summary": summarize_feature_groups(target_realized_features),
                "expected_delta_phi": serialize_feature_groups(expected_delta_phi),
                "context": {
                    "budget": int(args.budget),
                    "baseline_seed": int(args.baseline_seed),
                    "training_row_count": len(response_rows),
                    "portrait_source": portrait_source,
                    "surrogate_model": args.surrogate_model,
                    "bootstrap_models": int(args.bootstrap_models),
                    "kappa": float(args.kappa),
                    "surrogate_output_path": surrogate_output_path,
                },
                "portrait_source": portrait_source,
                "support_size": int(candidate.support_size),
                "l1_shift": float(np.abs(np.asarray(candidate.delta_q, dtype=np.float32)).sum()),
                "portrait_summary": summarize_portrait_shift(delta_phi, feature_label_map),
                "rationale": {
                    "donors": list(candidate.donors),
                    "receivers": list(candidate.receivers),
                    "amplitude": float(candidate.amplitude),
                    "pair_selector": args.pair_selector,
                    "plan": list(candidate.metadata.get("plan", [])),
                },
                "search_node_id": candidate.metadata.get("search_node_id"),
                "execution": {
                    "expected_slice_quotas": summarize_target_quotas(target_mixture, int(args.budget)),
                    "max_weight_sample_id": artifacts.sample_ids[int(np.argmax(weights))],
                    "selection_seed": selection_seed,
                    "selected_sample_ids": selected_ids,
                    "realized_mixture": [float(value) for value in np.asarray(materialization.realized_mixture, dtype=np.float32).tolist()],
                    "mixture_l1_before_coverage_repair": (
                        None if materialization.mixture_l1_before_coverage_repair is None
                        else float(materialization.mixture_l1_before_coverage_repair)
                    ),
                    "mixture_l1_after_coverage_repair": float(materialization.mixture_l1_after_coverage_repair),
                    "accepted_coverage_swaps": int(materialization.accepted_coverage_swaps),
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
    compact_ranked = [
        {
            "candidate_id": str(candidate.get("candidate_id", "")),
            "predicted_gain_mean": float(candidate.get("predicted_gain_mean", 0.0)),
            "predicted_gain_std": float(candidate.get("predicted_gain_std", 0.0)),
            "risk_adjusted_score": float(candidate.get("risk_adjusted_score", 0.0)),
            "delta_q": list(candidate.get("delta_q", [])),
            "rationale": dict(candidate.get("rationale", {})),
            "portrait_summary": dict(candidate.get("portrait_summary", {})),
        }
        for candidate in ranked
    ]
    ranked[0]["ranked_candidates"] = compact_ranked
    if search_tree_trace is not None:
        ranked[0]["search_tree"] = _attach_search_tree_metadata(
            trace=search_tree_trace,
            ranked=ranked,
            slice_ids=search_tree_slice_ids,
        )
    result = build_recommendation_result(ranked[0])

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
