from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from slice_discovery.projector import SliceFeatureProjector
    from slice_remix.baseline import estimate_baseline_mixture, load_slice_artifacts
    from slice_remix.beam_search import BeamSearchConfig, SearchEdge, generate_beam_candidates_with_trace
    from slice_remix.portraits import (
        build_feature_label_map,
        compute_portrait_shift,
        compute_slice_portraits,
        load_portrait_feature_groups,
        summarize_portrait_shift,
    )
    from slice_remix.prior_graph import (
        PriorGraphHyperparams,
        build_portrait_residual_context,
        build_prior_graph,
    )
else:
    from .slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from .slice_discovery.projector import SliceFeatureProjector
    from .slice_remix.baseline import estimate_baseline_mixture, load_slice_artifacts
    from .slice_remix.beam_search import BeamSearchConfig, SearchEdge, generate_beam_candidates_with_trace
    from .slice_remix.portraits import (
        build_feature_label_map,
        compute_portrait_shift,
        compute_slice_portraits,
        load_portrait_feature_groups,
        summarize_portrait_shift,
    )
    from .slice_remix.prior_graph import (
        PriorGraphHyperparams,
        build_portrait_residual_context,
        build_prior_graph,
    )


def _progress(message: str) -> None:
    print(f"[workbench_beam_export] {message}", file=sys.stderr, flush=True)


def _load_json(path: str | Path) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str | Path, payload: object) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _resolve_baseline_sample_indices(
    *,
    manifest_path: str | None,
    fallback_seed: int,
    fallback_budget: int,
    sample_ids: list[str],
) -> tuple[list[int], str]:
    sample_index = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    if manifest_path:
        manifest = _load_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError("baseline manifest must be a JSON object")
        selected_ids = [str(sample_id) for sample_id in manifest.get("sample_ids", [])]
        indices = [sample_index[sample_id] for sample_id in selected_ids if sample_id in sample_index]
        if not indices:
            raise ValueError(f"baseline manifest {manifest_path} did not resolve to any atlas sample ids")
        return indices, os.path.abspath(manifest_path)

    rng = np.random.default_rng(int(fallback_seed))
    indices = rng.choice(len(sample_ids), size=int(fallback_budget), replace=False).tolist()
    return indices, "rng_fallback"


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
                balance_score=float(edge.balance_score),
                risk_score=float(edge.risk_score),
                amplitude_band=tuple(float(value) for value in edge.amplitude_band),
            )
        )
    return edges


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


def _sanitize_feature_label(label: str) -> str:
    sanitized = label.replace("[", "_").replace("]", "").replace(".", "_").replace("-", "_")
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


def _extract_changed_fields(portrait_summary: dict[str, object], *, max_fields: int = 5) -> tuple[list[str], list[str]]:
    changed_blocks: list[str] = []
    changed_fields: list[str] = []
    for block in portrait_summary.get("top_blocks", []):
        if not isinstance(block, dict):
            continue
        block_name = str(block.get("block_name", ""))
        if not block_name:
            continue
        dimension = block_name.split(".", 1)[0]
        feature_prefix = block_name.split(".", 1)[1] if "." in block_name else block_name
        if dimension and dimension not in changed_blocks:
            changed_blocks.append(dimension)
        for feature in block.get("top_features", []):
            if not isinstance(feature, dict):
                continue
            field_name = _sanitize_feature_label(str(feature.get("feature", "")))
            if not field_name:
                continue
            token = _sanitize_feature_label(f"{feature_prefix}_{field_name}")
            if token and token not in changed_fields:
                changed_fields.append(token)
            if len(changed_fields) >= max_fields:
                break
        if len(changed_fields) >= max_fields:
            break
    return changed_blocks[:3], changed_fields[:max_fields]


def _expected_shift_summary(portrait_summary: dict[str, object]) -> str:
    fragments: list[str] = []
    for block in portrait_summary.get("top_blocks", [])[:3]:
        if not isinstance(block, dict):
            continue
        block_name = str(block.get("block_name", ""))
        if not block_name:
            continue
        dimension = block_name.split(".", 1)[0]
        signed_sum = float(block.get("signed_sum", 0.0))
        direction = "rises" if signed_sum >= 0.0 else "drops"
        fragments.append(f"{dimension} {direction}")
    if not fragments:
        return "Beam search suggests a small atlas rebalancing move."
    if len(fragments) == 1:
        return f"{fragments[0].capitalize()} under the current beam proposal."
    return f"{', '.join(fragments[:-1]).capitalize()}, and {fragments[-1]} under the current beam proposal."


def _build_candidate_rankings(
    *,
    candidates,
    trace: dict[str, object],
    portraits: dict[str, np.ndarray],
    feature_label_map: dict[str, list[str]],
    baseline_mixture: np.ndarray,
    slice_ids: list[str],
    round_id: int,
    candidate_limit: int,
    shuffle_seed: int,
) -> list[dict[str, object]]:
    search_node_lookup = {
        str(node.get("node_id", "")): dict(node)
        for node in trace.get("nodes", [])
        if isinstance(node, dict)
    }
    candidate_rows: list[dict[str, object]] = []
    for candidate in list(candidates)[: max(1, int(candidate_limit))]:
        target_mixture = np.asarray(candidate.target_mixture, dtype=np.float32)
        delta_phi = compute_portrait_shift(portraits, baseline_mixture, target_mixture)
        portrait_summary = summarize_portrait_shift(delta_phi, feature_label_map)
        changed_blocks, changed_fields = _extract_changed_fields(portrait_summary)
        plan = list(candidate.metadata.get("plan", []))
        transfer_pairs = [
            {
                "donor": slice_ids[int(step["donor"])],
                "receiver": slice_ids[int(step["receiver"])],
                "amplitude": float(step["amplitude"]),
            }
            for step in plan
        ]
        search_node_id = str(candidate.metadata.get("search_node_id", ""))
        node = search_node_lookup.get(search_node_id, {})
        candidate_rows.append(
            {
                "candidate_id": "",
                "transfer_pairs": transfer_pairs,
                "delta_q": [float(value) for value in np.asarray(candidate.delta_q, dtype=np.float32).tolist()],
                "predicted_gain_mean": float(node.get("progress", 0.0)),
                "predicted_gain_std": 0.0,
                "risk_adjusted_score": float(node.get("priority", 0.0)),
                "changed_blocks": changed_blocks,
                "changed_fields": changed_fields,
                "preference_alignment": "exploratory",
                "expected_shift_summary": _expected_shift_summary(portrait_summary),
                "search_node_id": search_node_id,
            }
        )

    if not candidate_rows:
        return []

    order = np.arange(len(candidate_rows), dtype=np.int64)
    np.random.default_rng(int(shuffle_seed)).shuffle(order)
    ranked: list[dict[str, object]] = []
    for rank_index, candidate_index in enumerate(order, start=1):
        row = dict(candidate_rows[int(candidate_index)])
        row["candidate_id"] = f"cand_r{round_id}_{rank_index}"
        ranked.append(row)
    return ranked


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export beam-only prior graph and recommendation rounds for the workbench.")
    parser.add_argument("--projected-dir", required=True)
    parser.add_argument("--cluster-dir", required=True)
    parser.add_argument("--schema-path", required=True)
    parser.add_argument("--input-bundle-root", required=True)
    parser.add_argument("--output-root", action="append", required=True)
    parser.add_argument("--portrait-source", choices=["auto", "projected", "semantic"], default="semantic")
    parser.add_argument("--processed-data-root")
    parser.add_argument("--assembled-feature-dir")
    parser.add_argument("--baseline-seed", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--baseline-manifest-path")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--top-k-render", type=int, default=12)
    parser.add_argument("--beam-max-depth", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--proposal-edges-per-node", type=int, default=12)
    parser.add_argument("--beam-stop-epsilon", type=float, default=1e-3)
    parser.add_argument("--beam-donor-keep-ratio", type=float, default=0.2)
    parser.add_argument("--beam-min-transfer-mass", type=float, default=0.03)
    parser.add_argument("--beam-receiver-headroom", type=float, default=0.15)
    parser.add_argument("--candidate-limit", type=int, default=12)
    return parser


def run(args: argparse.Namespace, log_fn=_progress) -> int:
    input_bundle_root = Path(args.input_bundle_root).resolve()
    task_context = _load_json(input_bundle_root / "task_context.json")
    if not isinstance(task_context, dict):
        raise ValueError("task_context.json must be a JSON object")

    baseline_seed = int(args.baseline_seed if args.baseline_seed is not None else task_context.get("baseline_seed", 0))
    budget = int(args.budget if args.budget is not None else task_context.get("baseline_budget", 0))
    round_count = int(task_context.get("round_count", 3))

    projected = SliceFeatureProjector.load(os.path.abspath(args.projected_dir))
    artifacts = load_slice_artifacts(os.path.abspath(args.cluster_dir))
    if projected.sample_ids != artifacts.sample_ids:
        raise ValueError("projected sample ids must match cluster sample ids")

    feature_groups, portrait_source = load_portrait_feature_groups(
        projected=projected,
        cluster_meta=artifacts.meta,
        portrait_source=args.portrait_source,
        processed_data_root=os.path.abspath(args.processed_data_root) if args.processed_data_root else None,
        schema_path=os.path.abspath(args.schema_path),
        assembled_feature_dir=os.path.abspath(args.assembled_feature_dir) if args.assembled_feature_dir else None,
        log_fn=log_fn,
    )
    feature_label_map = build_feature_label_map(
        feature_groups,
        schema_path=os.path.abspath(args.schema_path),
    )

    baseline_sample_indices, baseline_source = _resolve_baseline_sample_indices(
        manifest_path=args.baseline_manifest_path,
        fallback_seed=baseline_seed,
        fallback_budget=budget,
        sample_ids=artifacts.sample_ids,
    )
    log_fn(f"baseline sample source={baseline_source} size={len(baseline_sample_indices)}")

    baseline_mixture = estimate_baseline_mixture(artifacts.membership, baseline_sample_indices)
    pool_mixture = artifacts.membership.mean(axis=0, dtype=np.float32)
    slice_ids = _slice_ids(artifacts.membership.shape[1])

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        portraits = compute_slice_portraits(feature_groups, artifacts.membership)

        prior_graph_payload = build_prior_graph(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=artifacts.membership,
            baseline_sample_indices=baseline_sample_indices,
            slice_ids=slice_ids,
            hyperparams=PriorGraphHyperparams(
                top_k_render=int(args.top_k_render),
                score_threshold=float(args.score_threshold),
            ),
            baseline_seed=baseline_seed,
            budget=budget,
        )
        prior_graph = prior_graph_payload.to_dict()

        portrait_context = build_portrait_residual_context(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=artifacts.membership,
            baseline_sample_indices=baseline_sample_indices,
        )
        beam_candidates, beam_trace = generate_beam_candidates_with_trace(
            baseline_mixture=baseline_mixture,
            pool_mixture=pool_mixture,
            edges=_payload_to_search_edges(prior_graph_payload, slice_ids),
            portrait_context=portrait_context,
            config=BeamSearchConfig(
                max_depth=int(args.beam_max_depth),
                beam_width=int(args.beam_width),
                proposal_edges_per_node=int(args.proposal_edges_per_node),
                stop_epsilon=float(args.beam_stop_epsilon),
                donor_keep_ratio=float(args.beam_donor_keep_ratio),
                min_transfer_mass=float(args.beam_min_transfer_mass),
                receiver_headroom=float(args.beam_receiver_headroom),
            ),
        )

    prior_graph["graph_context"]["portrait_source"] = portrait_source
    prior_graph["graph_context"]["baseline_manifest_path"] = baseline_source

    round_payloads: dict[int, dict[str, object]] = {}
    for round_id in range(1, round_count + 1):
        source_path = input_bundle_root / f"recommendation_round_{round_id}.json"
        source_payload = _load_json(source_path) if source_path.exists() else {}
        if not isinstance(source_payload, dict):
            source_payload = {}
        ranked = _build_candidate_rankings(
            candidates=beam_candidates,
            trace=beam_trace,
            portraits=portraits,
            feature_label_map=feature_label_map,
            baseline_mixture=baseline_mixture,
            slice_ids=slice_ids,
            round_id=round_id,
            candidate_limit=int(args.candidate_limit),
            shuffle_seed=baseline_seed + round_id,
        )
        search_tree = _attach_search_tree_metadata(
            trace=beam_trace,
            ranked=ranked,
            slice_ids=slice_ids,
        )
        clean_ranked = []
        for row in ranked:
            clean_row = dict(row)
            clean_row.pop("search_node_id", None)
            clean_ranked.append(clean_row)

        round_payload = {
            "round_id": int(source_payload.get("round_id", round_id)),
            "baseline_id": source_payload.get("baseline_id", f"baseline_seed{baseline_seed}_b{budget}"),
            "hypothesis": source_payload.get("hypothesis", {}),
            "controls": source_payload.get("controls", {}),
            "candidate_rankings": clean_ranked,
            "search_tree": search_tree,
            "recommended_candidate_id": clean_ranked[0]["candidate_id"] if clean_ranked else None,
        }
        round_payloads[round_id] = round_payload

    for output_root in args.output_root:
        root = Path(output_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / "prior_graph.json", prior_graph)
        for round_id, payload in round_payloads.items():
            _write_json(root / f"recommendation_round_{round_id}.json", payload)
        log_fn(f"wrote beam workbench export to {root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
