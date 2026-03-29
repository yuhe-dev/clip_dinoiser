from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from run_workbench_beam_export import (
        _attach_search_tree_metadata,
        _build_candidate_rankings,
        _payload_to_search_edges,
        _resolve_baseline_sample_indices,
        _slice_ids,
        _write_json,
    )
    from slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from slice_discovery.projector import SliceFeatureProjector
    from slice_remix.baseline import estimate_baseline_mixture, load_slice_artifacts
    from slice_remix.beam_search import BeamSearchConfig, generate_beam_candidates_with_trace
    from slice_remix.dataset import write_jsonl
    from slice_remix.portraits import (
        build_feature_label_map,
        compute_expected_portrait,
        compute_slice_portraits,
        load_portrait_feature_groups,
    )
    from slice_remix.prior_graph import (
        PriorGraphHyperparams,
        build_portrait_residual_context,
        build_prior_graph,
    )
else:
    from .run_workbench_beam_export import (
        _attach_search_tree_metadata,
        _build_candidate_rankings,
        _payload_to_search_edges,
        _resolve_baseline_sample_indices,
        _slice_ids,
        _write_json,
    )
    from .slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from .slice_discovery.projector import SliceFeatureProjector
    from .slice_remix.baseline import estimate_baseline_mixture, load_slice_artifacts
    from .slice_remix.beam_search import BeamSearchConfig, generate_beam_candidates_with_trace
    from .slice_remix.dataset import write_jsonl
    from .slice_remix.portraits import (
        build_feature_label_map,
        compute_expected_portrait,
        compute_slice_portraits,
        load_portrait_feature_groups,
    )
    from .slice_remix.prior_graph import (
        PriorGraphHyperparams,
        build_portrait_residual_context,
        build_prior_graph,
    )


def _progress(message: str) -> None:
    print(f"[search_tree_census] {message}", file=sys.stderr, flush=True)


def _parse_int_csv(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def _normalize_mixture(values: np.ndarray) -> np.ndarray:
    mixture = np.asarray(values, dtype=np.float32).reshape(-1)
    mixture = np.clip(mixture, 0.0, None)
    total = float(mixture.sum())
    if total > 0.0:
        mixture = mixture / total
    return mixture.astype(np.float32)


def _candidate_target_mixture(baseline_mixture: np.ndarray, delta_q: list[float] | np.ndarray) -> np.ndarray:
    return _normalize_mixture(np.asarray(baseline_mixture, dtype=np.float32) + np.asarray(delta_q, dtype=np.float32))


def _resolve_seed_manifest_path(baseline_manifest_dir: str | None, baseline_seed: int) -> str | None:
    if not baseline_manifest_dir:
        return None
    root = Path(baseline_manifest_dir).resolve()
    candidates = [
        root / f"baseline_{baseline_seed}.json",
        root / f"seed_{baseline_seed}.json",
        root / f"{baseline_seed}.json",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _serialize_expected_portrait(expected: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {
        block_name: [float(value) for value in np.asarray(vector, dtype=np.float32).reshape(-1).tolist()]
        for block_name, vector in expected.items()
    }


def _summarize_expected_portrait(expected: dict[str, np.ndarray]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for block_name, vector in expected.items():
        values = np.asarray(vector, dtype=np.float32).reshape(-1)
        summary[block_name] = {
            "dimension": int(values.shape[0]),
            "mean": float(values.mean()) if values.size else 0.0,
            "std": float(values.std()) if values.size else 0.0,
            "min": float(values.min()) if values.size else 0.0,
            "max": float(values.max()) if values.size else 0.0,
            "l1_norm": float(np.abs(values).sum()) if values.size else 0.0,
            "l2_norm": float(np.linalg.norm(values)) if values.size else 0.0,
        }
    return summary


def _primitive_pair_tokens(transfer_pairs: list[dict[str, object]]) -> list[str]:
    tokens: list[str] = []
    for step in transfer_pairs:
        donor = str(step.get("donor", "")).strip()
        receiver = str(step.get("receiver", "")).strip()
        if donor and receiver:
            tokens.append(f"{donor}->{receiver}")
    return tokens


def _baseline_mixture_dispersion(baseline_mixtures: list[np.ndarray]) -> dict[str, float | int]:
    if len(baseline_mixtures) < 2:
        return {
            "pair_count": 0,
            "mean_l1": 0.0,
            "min_l1": 0.0,
            "max_l1": 0.0,
        }

    distances: list[float] = []
    for left_index in range(len(baseline_mixtures)):
        for right_index in range(left_index + 1, len(baseline_mixtures)):
            left = np.asarray(baseline_mixtures[left_index], dtype=np.float32)
            right = np.asarray(baseline_mixtures[right_index], dtype=np.float32)
            distances.append(float(np.abs(left - right).sum()))
    return {
        "pair_count": len(distances),
        "mean_l1": float(np.mean(distances)),
        "min_l1": float(np.min(distances)),
        "max_l1": float(np.max(distances)),
    }


def _candidate_row_for_pool(
    *,
    baseline_seed: int,
    budget: int,
    baseline_trial_id: str,
    baseline_mixture: np.ndarray,
    baseline_expected: dict[str, np.ndarray],
    candidate_rank: int,
    candidate_row: dict[str, object],
    recommended_candidate_id: str | None,
    portraits: dict[str, np.ndarray],
) -> dict[str, object]:
    target_mixture = _candidate_target_mixture(baseline_mixture, candidate_row.get("delta_q", []))
    target_expected = compute_expected_portrait(portraits, target_mixture)
    transfer_pairs = list(candidate_row.get("transfer_pairs", []))
    candidate_id = str(candidate_row.get("candidate_id", ""))
    return {
        "baseline_seed": int(baseline_seed),
        "baseline_budget": int(budget),
        "baseline_trial_id": baseline_trial_id,
        "candidate_id": candidate_id,
        "candidate_rank": int(candidate_rank),
        "is_recommended": bool(candidate_id and candidate_id == recommended_candidate_id),
        "baseline_mixture": [float(value) for value in np.asarray(baseline_mixture, dtype=np.float32).tolist()],
        "target_mixture": [float(value) for value in np.asarray(target_mixture, dtype=np.float32).tolist()],
        "transfer_pairs": transfer_pairs,
        "plan_length": int(len(transfer_pairs)),
        "primitive_pair_count": int(len(transfer_pairs)),
        "baseline_features_raw": _serialize_expected_portrait(baseline_expected),
        "target_features_raw": _serialize_expected_portrait(target_expected),
        "baseline_features_summary": _summarize_expected_portrait(baseline_expected),
        "target_features_summary": _summarize_expected_portrait(target_expected),
        "analysis_only": {
            "delta_q": list(candidate_row.get("delta_q", [])),
            "predicted_gain_mean": float(candidate_row.get("predicted_gain_mean", 0.0)),
            "predicted_gain_std": float(candidate_row.get("predicted_gain_std", 0.0)),
            "risk_adjusted_score": float(candidate_row.get("risk_adjusted_score", 0.0)),
            "changed_blocks": list(candidate_row.get("changed_blocks", [])),
            "changed_fields": list(candidate_row.get("changed_fields", [])),
            "preference_alignment": candidate_row.get("preference_alignment"),
            "expected_shift_summary": candidate_row.get("expected_shift_summary"),
        },
    }


def _build_seed_summary(
    *,
    baseline_seed: int,
    baseline_trial_id: str,
    baseline_source: str,
    baseline_sample_ids: list[str],
    baseline_mixture: np.ndarray,
    search_tree: dict[str, object],
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    node_depth_hist: Counter[int] = Counter()
    completed_nodes = 0
    for node in search_tree.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_depth_hist[int(node.get("depth", 0))] += 1
        if str(node.get("node_type", "")) == "completed":
            completed_nodes += 1

    candidate_plan_length_hist: Counter[int] = Counter()
    unique_pairs: set[str] = set()
    for row in candidates:
        plan_length = int(row.get("plan_length", 0))
        candidate_plan_length_hist[plan_length] += 1
        unique_pairs.update(_primitive_pair_tokens(list(row.get("transfer_pairs", []))))

    return {
        "baseline_seed": int(baseline_seed),
        "baseline_trial_id": baseline_trial_id,
        "baseline_source": baseline_source,
        "baseline_sample_size": int(len(baseline_sample_ids)),
        "baseline_sample_ids": list(baseline_sample_ids),
        "baseline_mixture": [float(value) for value in np.asarray(baseline_mixture, dtype=np.float32).tolist()],
        "search_node_count": int(len(search_tree.get("nodes", []))),
        "completed_node_count": int(completed_nodes),
        "candidate_count": int(len(candidates)),
        "multi_step_candidate_count": int(sum(1 for row in candidates if int(row.get("plan_length", 0)) >= 2)),
        "node_depth_hist": {str(depth): int(count) for depth, count in sorted(node_depth_hist.items())},
        "candidate_plan_length_hist": {str(depth): int(count) for depth, count in sorted(candidate_plan_length_hist.items())},
        "unique_primitive_pairs_top_candidates": sorted(unique_pairs),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-run single-round beam search sessions across many baseline seeds.")
    parser.add_argument("--projected-dir", required=True)
    parser.add_argument("--cluster-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--baseline-seeds", default="0")
    parser.add_argument("--baseline-manifest-dir")
    parser.add_argument("--portrait-source", choices=["auto", "projected", "semantic"], default="auto")
    parser.add_argument("--processed-data-root")
    parser.add_argument("--schema-path")
    parser.add_argument("--assembled-feature-dir")
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
    baseline_seeds = _parse_int_csv(args.baseline_seeds)
    if not baseline_seeds:
        raise ValueError("baseline-seeds must contain at least one integer")

    projected = SliceFeatureProjector.load(os.path.abspath(args.projected_dir))
    artifacts = load_slice_artifacts(os.path.abspath(args.cluster_dir))
    if projected.sample_ids != artifacts.sample_ids:
        raise ValueError("projected sample ids must match cluster sample ids")

    feature_groups, portrait_source = load_portrait_feature_groups(
        projected=projected,
        cluster_meta=artifacts.meta,
        portrait_source=args.portrait_source,
        processed_data_root=os.path.abspath(args.processed_data_root) if args.processed_data_root else None,
        schema_path=os.path.abspath(args.schema_path) if args.schema_path else None,
        assembled_feature_dir=os.path.abspath(args.assembled_feature_dir) if args.assembled_feature_dir else None,
        log_fn=log_fn,
    )
    feature_label_map = build_feature_label_map(
        feature_groups,
        schema_path=os.path.abspath(args.schema_path) if args.schema_path else None,
    )
    portraits = compute_slice_portraits(feature_groups, artifacts.membership)
    pool_mixture = artifacts.membership.mean(axis=0, dtype=np.float32)
    slice_ids = _slice_ids(artifacts.membership.shape[1])

    output_root = Path(args.output_root).resolve()
    sessions_root = output_root / "sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "feature_label_map.json", feature_label_map)

    candidate_pool_rows: list[dict[str, object]] = []
    session_index: list[dict[str, object]] = []
    seed_level_summary: list[dict[str, object]] = []
    node_depth_hist: Counter[int] = Counter()
    candidate_plan_length_hist: Counter[int] = Counter()
    primitive_pair_support: dict[str, set[int]] = defaultdict(set)
    baseline_mixtures: list[np.ndarray] = []
    total_search_nodes = 0

    for baseline_seed in baseline_seeds:
        manifest_path = _resolve_seed_manifest_path(args.baseline_manifest_dir, baseline_seed)
        baseline_sample_indices, baseline_source = _resolve_baseline_sample_indices(
            manifest_path=manifest_path,
            fallback_seed=baseline_seed,
            fallback_budget=int(args.budget),
            sample_ids=artifacts.sample_ids,
        )
        baseline_sample_ids = [artifacts.sample_ids[int(index)] for index in baseline_sample_indices]
        baseline_mixture = estimate_baseline_mixture(artifacts.membership, baseline_sample_indices)
        baseline_mixtures.append(np.asarray(baseline_mixture, dtype=np.float32))
        baseline_trial_id = f"baseline_seed{baseline_seed}_b{int(args.budget)}"
        log_fn(
            f"baseline_seed={baseline_seed} sample_size={len(baseline_sample_indices)} "
            f"source={baseline_source}"
        )

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
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
                baseline_seed=int(baseline_seed),
                budget=int(args.budget),
            )
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

        prior_graph = prior_graph_payload.to_dict()
        prior_graph["graph_context"]["portrait_source"] = portrait_source
        prior_graph["graph_context"]["baseline_manifest_path"] = baseline_source

        ranked = _build_candidate_rankings(
            candidates=beam_candidates,
            trace=beam_trace,
            portraits=portraits,
            feature_label_map=feature_label_map,
            baseline_mixture=baseline_mixture,
            slice_ids=slice_ids,
            round_id=1,
            candidate_limit=int(args.candidate_limit),
            shuffle_seed=int(baseline_seed) + 1,
        )
        for candidate_rank, row in enumerate(ranked, start=1):
            row["candidate_id"] = f"seed{baseline_seed}_cand_{candidate_rank:02d}"

        search_tree = _attach_search_tree_metadata(
            trace=beam_trace,
            ranked=ranked,
            slice_ids=slice_ids,
        )
        search_tree["layer_summaries"] = list(beam_trace.get("layer_summaries", []))
        clean_ranked: list[dict[str, object]] = []
        for row in ranked:
            clean_row = dict(row)
            clean_row.pop("search_node_id", None)
            clean_ranked.append(clean_row)

        baseline_expected = compute_expected_portrait(portraits, baseline_mixture)
        session_candidate_rows: list[dict[str, object]] = []
        recommended_candidate_id = clean_ranked[0]["candidate_id"] if clean_ranked else None
        for candidate_rank, candidate_row in enumerate(clean_ranked, start=1):
            pool_row = _candidate_row_for_pool(
                baseline_seed=int(baseline_seed),
                budget=int(args.budget),
                baseline_trial_id=baseline_trial_id,
                baseline_mixture=baseline_mixture,
                baseline_expected=baseline_expected,
                candidate_rank=candidate_rank,
                candidate_row=candidate_row,
                recommended_candidate_id=recommended_candidate_id,
                portraits=portraits,
            )
            candidate_pool_rows.append(pool_row)
            session_candidate_rows.append(pool_row)
            candidate_plan_length_hist[int(pool_row["plan_length"])] += 1
            for token in _primitive_pair_tokens(list(pool_row["transfer_pairs"])):
                primitive_pair_support[token].add(int(baseline_seed))

        seed_summary = _build_seed_summary(
            baseline_seed=int(baseline_seed),
            baseline_trial_id=baseline_trial_id,
            baseline_source=baseline_source,
            baseline_sample_ids=baseline_sample_ids,
            baseline_mixture=baseline_mixture,
            search_tree=search_tree,
            candidates=session_candidate_rows,
        )
        seed_level_summary.append(seed_summary)
        total_search_nodes += int(seed_summary["search_node_count"])
        for depth, count in seed_summary["node_depth_hist"].items():
            node_depth_hist[int(depth)] += int(count)

        session_dir = sessions_root / f"baseline_seed_{int(baseline_seed):04d}"
        session_dir.mkdir(parents=True, exist_ok=True)
        session_payload = {
            "baseline_seed": int(baseline_seed),
            "baseline_budget": int(args.budget),
            "baseline_trial_id": baseline_trial_id,
            "baseline_source": baseline_source,
            "baseline_sample_ids": baseline_sample_ids,
            "baseline_mixture": [float(value) for value in np.asarray(baseline_mixture, dtype=np.float32).tolist()],
            "pool_mixture": [float(value) for value in np.asarray(pool_mixture, dtype=np.float32).tolist()],
            "recommended_candidate_id": recommended_candidate_id,
            "candidate_count": int(len(session_candidate_rows)),
            "search_node_count": int(len(search_tree.get("nodes", []))),
            "portrait_source": portrait_source,
        }
        _write_json(session_dir / "session.json", session_payload)
        _write_json(session_dir / "prior_graph.json", prior_graph)
        _write_json(session_dir / "search_tree.json", search_tree)
        _write_json(session_dir / "completed_candidates.json", session_candidate_rows)

        session_index.append(
            {
                "baseline_seed": int(baseline_seed),
                "baseline_trial_id": baseline_trial_id,
                "session_dir": str(session_dir),
                "session_path": str(session_dir / "session.json"),
                "prior_graph_path": str(session_dir / "prior_graph.json"),
                "search_tree_path": str(session_dir / "search_tree.json"),
                "completed_candidates_path": str(session_dir / "completed_candidates.json"),
            }
        )

    unique_pairs = sorted(primitive_pair_support.keys())
    recurrent_pairs = sorted(
        token for token, seeds in primitive_pair_support.items() if len(seeds) >= 2
    )
    global_coverage_report = {
        "seed_count": int(len(baseline_seeds)),
        "candidate_pool_size": int(len(candidate_pool_rows)),
        "total_search_nodes": int(total_search_nodes),
        "total_completed_candidates": int(len(candidate_pool_rows)),
        "node_depth_hist": {str(depth): int(count) for depth, count in sorted(node_depth_hist.items())},
        "candidate_plan_length_hist": {str(depth): int(count) for depth, count in sorted(candidate_plan_length_hist.items())},
        "unique_primitive_pair_count_top_candidates": int(len(unique_pairs)),
        "unique_primitive_pairs_top_candidates": unique_pairs,
        "recurrent_primitive_pair_count_top_candidates": int(len(recurrent_pairs)),
        "recurrent_primitive_pairs_top_candidates": recurrent_pairs,
        "baseline_mixture_dispersion": _baseline_mixture_dispersion(baseline_mixtures),
    }

    write_jsonl(str(output_root / "candidate_pool_unlabeled.jsonl"), candidate_pool_rows)
    _write_json(output_root / "session_index.json", session_index)
    _write_json(output_root / "seed_level_summary.json", seed_level_summary)
    _write_json(output_root / "global_coverage_report.json", global_coverage_report)
    log_fn(
        f"wrote search-tree census to {output_root} seeds={len(baseline_seeds)} "
        f"candidate_pool={len(candidate_pool_rows)}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
