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
    from slice_remix.baseline import load_slice_artifacts
    from slice_remix.beam_search import (
        SearchEdge,
        TargetBeamSearchConfig,
        generate_target_beam_candidates_with_trace,
    )
    from slice_remix.portraits import build_feature_label_map, load_portrait_feature_groups
    from slice_remix.prior_graph import (
        PriorGraphHyperparams,
        SearchBias,
        SearchConstraints,
        TargetPortraitSpec,
        build_pool_target_portrait_spec,
        build_target_prior_graph,
        build_target_residual_context,
        compute_target_residual_gap,
    )
else:
    from .slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from .slice_discovery.projector import SliceFeatureProjector
    from .slice_remix.baseline import load_slice_artifacts
    from .slice_remix.beam_search import (
        SearchEdge,
        TargetBeamSearchConfig,
        generate_target_beam_candidates_with_trace,
    )
    from .slice_remix.portraits import build_feature_label_map, load_portrait_feature_groups
    from .slice_remix.prior_graph import (
        PriorGraphHyperparams,
        SearchBias,
        SearchConstraints,
        TargetPortraitSpec,
        build_pool_target_portrait_spec,
        build_target_prior_graph,
        build_target_residual_context,
        compute_target_residual_gap,
    )


def build_parser() -> argparse.ArgumentParser:
    default_schema = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "docs",
        "feature_schema",
        "unified_processed_feature_schema.json",
    )
    parser = argparse.ArgumentParser(description="Run explicit-target prior graph and beam-search review on a real slice atlas.")
    parser.add_argument("--cluster-dir", default="./clip_dinoiser/artifacts/slice_canonical_vmf_auto_k24")
    parser.add_argument("--projected-dir")
    parser.add_argument("--portrait-source", choices=["auto", "projected", "semantic"], default="projected")
    parser.add_argument("--processed-data-root")
    parser.add_argument("--schema-path", default=default_schema)
    parser.add_argument("--assembled-feature-dir")
    parser.add_argument("--baseline-seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k-render", type=int, default=16)
    parser.add_argument("--score-threshold", type=float, default=-2.0)
    parser.add_argument("--beam-max-depth", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--proposal-edges-per-node", type=int, default=24)
    parser.add_argument("--beam-stop-epsilon", type=float, default=1e-3)
    parser.add_argument("--target-mode", choices=["raw_pool", "quality_laplacian_shift"], default="raw_pool")
    parser.add_argument("--target-shift-mass", type=float, default=0.08)
    parser.add_argument("--candidate-limit", type=int, default=12)
    return parser


def _progress(message: str) -> None:
    print(f"[target_search_review] {message}", file=sys.stderr, flush=True)


def _resolve_projected_dir(cluster_dir: str, explicit_projected_dir: str | None) -> str:
    if explicit_projected_dir:
        return os.path.abspath(explicit_projected_dir)
    meta_path = os.path.join(cluster_dir, "slice_result_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    projected_dir = str(meta.get("projected_dir", "")).strip()
    if not projected_dir:
        raise ValueError("projected_dir missing from slice_result_meta.json; pass --projected-dir explicitly")
    return os.path.abspath(projected_dir)


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


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _serialize_prior_graph(payload: object) -> dict[str, object]:
    return {
        "nodes": [_json_ready(asdict(node)) for node in payload.nodes],
        "edges": [_json_ready(asdict(edge)) for edge in payload.edges],
        "graph_context": _json_ready(payload.graph_context),
        "defaults": _json_ready(payload.defaults),
    }


def _edge_summary(edge) -> str:
    components = getattr(edge, "risk_components", {}) or {}
    return (
        f"- `{edge.donor} -> {edge.receiver}` | score={edge.score:.4f}, fit={edge.fit_score:.4f}, "
        f"bias={getattr(edge, 'bias_score', 0.0):.4f}, risk={edge.risk_score:.4f}, "
        f"side={float(components.get('side_risk', 0.0)):.4f}, "
        f"boundary={float(components.get('boundary_risk', 0.0)):.4f}, "
        f"support_empty={float(components.get('support_empty_risk', 0.0)):.4f}, "
        f"band=({edge.amplitude_band[0]:.3f}, {edge.amplitude_band[1]:.3f})"
    )


def _layer_assessment(layer: dict[str, object]) -> str:
    expanded = int(layer.get("expanded_children", 0))
    beam_out = int(layer.get("beam_out", 0))
    deduped = int(layer.get("deduped_children", 0))
    stopped = layer.get("stopped")
    if stopped == "no_children":
        return "该层没有生成任何可行 child，说明搜索已自然终止。"
    if stopped == "stop_epsilon":
        return "该层存在 child，但最佳 child 的优先级提升不足以越过停止阈值。"
    if expanded <= max(1, beam_out):
        return "该层展开偏窄，proposal frontier 基本没有提供太多分支。"
    if deduped > max(beam_out * 4, 12):
        return "该层展开偏宽，dedup 后仍保留了较多状态，后续可继续收紧 proposal。"
    return "该层展开宽度处于可接受范围，既没有明显过窄，也没有明显爆炸。"


def _candidate_review_lines(candidates: list, limit: int) -> list[str]:
    lines: list[str] = []
    for index, candidate in enumerate(candidates[:limit], start=1):
        metadata = dict(candidate.metadata)
        plan = metadata.get("plan", [])
        lines.append(
            f"- 候选 {index}: progress={float(metadata.get('progress', 0.0)):.4f}, "
            f"opportunity={float(metadata.get('opportunity', 0.0)):.4f}, "
            f"complexity={float(metadata.get('complexity', 0.0)):.4f}, "
            f"priority={float(metadata.get('search_priority', 0.0)):.4f}, "
            f"support={int(candidate.support_size)}, "
            f"plan={json.dumps(plan, ensure_ascii=False)}"
        )
    if not lines:
        lines.append("- 没有生成任何 completed candidate。")
    return lines


def run(args: argparse.Namespace, log_fn=_progress) -> int:
    cluster_dir = os.path.abspath(args.cluster_dir)
    projected_dir = _resolve_projected_dir(cluster_dir, args.projected_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_fn("loading projected artifacts")
    projected = SliceFeatureProjector.load(projected_dir)
    log_fn("loading slice artifacts")
    artifacts = load_slice_artifacts(cluster_dir)
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
    feature_label_map = build_feature_label_map(
        feature_groups,
        schema_path=os.path.abspath(args.schema_path) if args.schema_path else None,
    )

    rng = np.random.default_rng(int(args.baseline_seed))
    sample_indices = rng.choice(len(artifacts.sample_ids), size=int(args.budget), replace=False)
    baseline_sample_indices = sample_indices.tolist()

    log_fn("building explicit target portrait")
    pool_target = build_pool_target_portrait_spec(
        feature_groups=feature_groups,
        feature_label_map=feature_label_map,
        memberships=artifacts.membership,
    )
    if args.target_mode == "quality_laplacian_shift":
        target_spec = _shift_quality_laplacian_target(
            pool_target,
            mass=float(args.target_shift_mass),
        )
    else:
        target_spec = pool_target
    target_context = build_target_residual_context(
        feature_groups=feature_groups,
        feature_label_map=feature_label_map,
        memberships=artifacts.membership,
        baseline_sample_indices=baseline_sample_indices,
        target_spec=target_spec,
    )
    baseline_gap = compute_target_residual_gap(
        context=target_context,
        mixture=target_context.baseline_mixture,
    )

    log_fn("building target prior graph")
    payload = build_target_prior_graph(
        feature_groups=feature_groups,
        feature_label_map=feature_label_map,
        memberships=artifacts.membership,
        baseline_sample_indices=baseline_sample_indices,
        slice_ids=_slice_ids(artifacts.membership.shape[1]),
        target_spec=target_spec,
        target_context=target_context,
        constraints=SearchConstraints(),
        bias=SearchBias(),
        hyperparams=PriorGraphHyperparams(
            top_k_render=int(args.top_k_render),
            score_threshold=float(args.score_threshold),
        ),
        baseline_seed=int(args.baseline_seed),
        budget=int(args.budget),
    )
    prior_graph_json = _serialize_prior_graph(payload)
    prior_graph_json["graph_context"]["portrait_source"] = portrait_source
    prior_graph_json["graph_context"]["target_spec"] = {
        "source": target_spec.source,
        "mode": str(args.target_mode),
        "shape_targets": _json_ready(target_spec.shape_targets),
        "scalar_targets": _json_ready(target_spec.scalar_targets),
        "block_weights": _json_ready(target_spec.block_weights),
        "baseline_gap": float(baseline_gap),
    }

    prior_graph_path = os.path.join(output_dir, "prior_graph_target_v1.json")
    with open(prior_graph_path, "w", encoding="utf-8") as f:
        json.dump(prior_graph_json, f, indent=2, ensure_ascii=False)

    log_fn("running target beam search")
    candidates, trace = generate_target_beam_candidates_with_trace(
        baseline_mixture=target_context.baseline_mixture,
        edges=_payload_to_search_edges(payload, _slice_ids(artifacts.membership.shape[1])),
        target_context=target_context,
        config=TargetBeamSearchConfig(
            max_depth=int(args.beam_max_depth),
            beam_width=int(args.beam_width),
            proposal_edges_per_node=int(args.proposal_edges_per_node),
            stop_epsilon=float(args.beam_stop_epsilon),
        ),
    )
    beam_payload = {
        "trace": _json_ready(trace),
        "candidates": [_json_ready(asdict(candidate)) for candidate in candidates],
    }
    beam_trace_path = os.path.join(output_dir, "beam_target_v1_trace.json")
    with open(beam_trace_path, "w", encoding="utf-8") as f:
        json.dump(beam_payload, f, indent=2, ensure_ascii=False)

    admissible_edges = [edge for edge in payload.edges if edge.admissible]
    top_edges = sorted(admissible_edges, key=lambda edge: edge.score, reverse=True)[: min(12, len(admissible_edges))]
    highest_risk = sorted(admissible_edges, key=lambda edge: edge.risk_score, reverse=True)[: min(6, len(admissible_edges))]
    lowest_risk = sorted(admissible_edges, key=lambda edge: edge.risk_score)[: min(6, len(admissible_edges))]
    layer_summaries = list(trace.get("layer_summaries", []))

    review_lines = [
        "# 显式 Target Portrait 版 Prior Graph / Beam Search 审查",
        "",
        "## 运行配置",
        f"- atlas: `{cluster_dir}`",
        f"- projected_dir: `{projected_dir}`",
        f"- portrait_source: `{portrait_source}`",
        f"- baseline_seed: `{int(args.baseline_seed)}`",
        f"- budget: `{int(args.budget)}`",
        f"- target_mode: `{args.target_mode}`",
        f"- target block: `quality.laplacian`",
        f"- target shift mass: `{float(args.target_shift_mass):.3f}`",
        f"- baseline gap to target: `{float(baseline_gap):.6f}`",
        "",
        "## Target 编辑摘要",
        f"- target source: `{target_spec.source}`",
        f"- shape target bins: `{json.dumps(_json_ready(target_spec.shape_targets['quality.laplacian']))}`",
        "",
        "## Prior Graph Top Edges",
    ]
    review_lines.extend(_edge_summary(edge) for edge in top_edges)
    review_lines.extend(
        [
            "",
            "## Risk 审查",
            "### 风险最高的边",
        ]
    )
    review_lines.extend(_edge_summary(edge) for edge in highest_risk)
    review_lines.extend(
        [
            "",
            "### 风险最低的边",
        ]
    )
    review_lines.extend(_edge_summary(edge) for edge in lowest_risk)
    review_lines.extend(
        [
            "",
            "## Beam Search 层级统计",
        ]
    )
    if layer_summaries:
        for layer in layer_summaries:
            review_lines.append(
                f"- depth {int(layer.get('depth', 0))}: beam_in={int(layer.get('beam_in', 0))}, "
                f"expanded={int(layer.get('expanded_children', 0))}, "
                f"deduped={int(layer.get('deduped_children', 0))}, "
                f"beam_out={int(layer.get('beam_out', 0))}, "
                f"stopped={layer.get('stopped')}, "
                f"pruned={json.dumps(layer.get('pruned_summary', {}), ensure_ascii=False)}"
            )
            review_lines.append(f"  评语：{_layer_assessment(layer)}")
    else:
        review_lines.append("- 没有导出层级统计。")
    review_lines.extend(
        [
            "",
            "## Candidate 审查",
        ]
    )
    review_lines.extend(_candidate_review_lines(candidates, int(args.candidate_limit)))
    review_lines.extend(
        [
            "",
            "## 初步结论",
            f"- prior graph admissible edges 数量: `{len(admissible_edges)}`",
            f"- beam completed candidates 数量: `{len(candidates)}`",
            "- 本轮只审查算法是否沿显式 target 方向产生合理边和合理搜索展开，不评估真实训练性能。",
            f"- 完整 prior graph JSON: `{prior_graph_path}`",
            f"- 完整 beam trace JSON: `{beam_trace_path}`",
        ]
    )

    review_path = os.path.join(output_dir, "review_target_v1.md")
    with open(review_path, "w", encoding="utf-8") as f:
        f.write("\n".join(review_lines))
        f.write("\n")
    log_fn(f"wrote prior graph review to {review_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
