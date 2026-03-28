from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .prior_graph import (
    PortraitResidualContext,
    TargetResidualContext,
    compute_portrait_residual_gap,
    compute_target_residual_gap,
)
from .types import CandidateAction


STATE_DEDUP_EPSILON = 1e-4
AMPLITUDE_DEDUP_EPSILON = 1e-6
OPPORTUNITY_TOP_K = 3


@dataclass(frozen=True)
class SearchEdge:
    donor: int
    receiver: int
    score: float
    amplitude_band: tuple[float, float]
    balance_score: float = 0.0
    risk_score: float = 0.0
    fit_score: float = 0.0
    bias_score: float = 0.0


@dataclass(frozen=True)
class BeamSearchConfig:
    max_depth: int = 4
    beam_width: int = 8
    proposal_edges_per_node: int = 12
    lambda_opportunity: float = 0.25
    stop_epsilon: float = 1e-3
    donor_keep_ratio: float = 0.2
    min_transfer_mass: float = 0.03
    adaptive_min_transfer_quantile: float = 0.10
    adaptive_min_transfer_floor_min: float = 1e-4
    receiver_headroom: float = 0.15


@dataclass(frozen=True)
class TargetBeamSearchConfig:
    max_depth: int = 4
    beam_width: int = 8
    proposal_edges_per_node: int = 12
    lambda_opportunity: float = 0.05
    stop_epsilon: float = 1e-3
    donor_keep_ratio: float = 0.2
    min_transfer_mass: float = 0.03
    adaptive_min_transfer_quantile: float = 0.10
    adaptive_min_transfer_floor_min: float = 1e-4
    receiver_headroom: float = 0.15


@dataclass
class _SearchNode:
    state_key: tuple[int, ...]
    parent_key: tuple[int, ...] | None
    depth: int
    action: dict[str, float | int] | None
    mixture: np.ndarray
    delta_q: np.ndarray
    plan: list[dict[str, float | int]]
    priority: float
    progress: float
    opportunity: float
    complexity: float
    last_edge: tuple[int, int] | None
    pruned_summary: dict[str, int] = field(default_factory=dict)
    expanded: bool = False


def get_adaptive_amplitudes(
    *,
    delta_bal: float,
    delta_max: float,
    min_transfer_mass: float,
    epsilon: float = AMPLITUDE_DEDUP_EPSILON,
) -> list[float]:
    max_value = float(delta_max)
    if max_value + epsilon < float(min_transfer_mass):
        return []

    safe_floor = float(min_transfer_mass)
    safe_bal = max(float(delta_bal), safe_floor)
    small = max(safe_floor, float(np.sqrt(safe_floor * safe_bal)))
    medium = min(max_value, max(safe_floor, float(delta_bal)))
    raw = sorted([small, medium, max_value])

    deduped: list[float] = []
    for value in raw:
        if not deduped or abs(value - deduped[-1]) > epsilon:
            deduped.append(float(value))
    return deduped


def get_depth_adaptive_amplitudes(
    *,
    delta_max: float,
    min_transfer_mass: float,
    best_amplitude: float,
    depth: int,
    epsilon: float = AMPLITUDE_DEDUP_EPSILON,
) -> list[float]:
    max_value = float(delta_max)
    if max_value + epsilon < float(min_transfer_mass):
        return []

    lower = float(min_transfer_mass)
    best_value = float(np.clip(best_amplitude, lower, max_value))
    raw = [lower, max_value]
    if depth >= 1:
        raw.append(best_value)
    if depth >= 2:
        raw.extend([(lower + best_value) / 2.0, (best_value + max_value) / 2.0])
    if depth >= 3:
        raw.extend(
            [
                (lower + (lower + best_value) / 2.0) / 2.0,
                (((best_value + max_value) / 2.0) + max_value) / 2.0,
            ]
        )

    deduped: list[float] = []
    for value in sorted(raw):
        if not deduped or abs(value - deduped[-1]) > epsilon:
            deduped.append(float(value))
    return deduped


def generate_beam_candidates(
    *,
    baseline_mixture: np.ndarray,
    pool_mixture: np.ndarray,
    edges: list[SearchEdge],
    portrait_context: PortraitResidualContext | None = None,
    config: BeamSearchConfig | None = None,
) -> list[CandidateAction]:
    candidates, _trace = generate_beam_candidates_with_trace(
        baseline_mixture=baseline_mixture,
        pool_mixture=pool_mixture,
        edges=edges,
        portrait_context=portrait_context,
        config=config,
    )
    return candidates


def generate_beam_candidates_with_trace(
    *,
    baseline_mixture: np.ndarray,
    pool_mixture: np.ndarray,
    edges: list[SearchEdge],
    portrait_context: PortraitResidualContext | None = None,
    config: BeamSearchConfig | None = None,
) -> tuple[list[CandidateAction], dict[str, object]]:
    params = config or BeamSearchConfig()
    baseline = np.asarray(baseline_mixture, dtype=np.float32)
    pool = np.asarray(pool_mixture, dtype=np.float32)
    if baseline.ndim != 1 or pool.ndim != 1 or baseline.shape != pool.shape:
        raise ValueError("baseline_mixture and pool_mixture must be same-length 1D arrays")

    baseline_gap = (
        float(portrait_context.baseline_gap)
        if portrait_context is not None
        else _gap(baseline, pool)
    )
    donor_floors = (baseline * float(params.donor_keep_ratio)).astype(np.float32)
    receiver_caps = np.minimum(
        np.ones_like(pool, dtype=np.float32),
        pool + float(params.receiver_headroom),
    ).astype(np.float32)

    root = _make_node(
        mixture=baseline,
        baseline=baseline,
        pool=pool,
        plan=[],
        last_edge=None,
        edges=edges,
        baseline_gap=baseline_gap,
        portrait_context=portrait_context,
        donor_floors=donor_floors,
        receiver_caps=receiver_caps,
        config=params,
        depth=0,
        parent_key=None,
        action=None,
    )
    beam = [root]
    completed: list[_SearchNode] = []
    retained: dict[tuple[int, ...], _SearchNode] = {root.state_key: root}
    layer_summaries: list[dict[str, object]] = []

    for _depth in range(params.max_depth):
        beam_in_count = int(len(beam))
        expanded: list[_SearchNode] = []
        best_parent_progress = max(node.progress for node in beam) if beam else 0.0
        layer_pruned_totals: dict[str, int] = {}
        for node in beam:
            children, pruned_summary = _expand_node(
                node=node,
                baseline=baseline,
                pool=pool,
                edges=edges,
                baseline_gap=baseline_gap,
                portrait_context=portrait_context,
                donor_floors=donor_floors,
                receiver_caps=receiver_caps,
                config=params,
            )
            node.expanded = True
            node.pruned_summary = pruned_summary
            retained[node.state_key] = node
            expanded.extend(children)
            for reason, count in pruned_summary.items():
                layer_pruned_totals[reason] = layer_pruned_totals.get(reason, 0) + int(count)

        if not expanded:
            layer_summaries.append(
                {
                    "depth": int(_depth),
                    "beam_in": beam_in_count,
                    "expanded_children": 0,
                    "deduped_children": 0,
                    "beam_out": 0,
                    "best_parent_progress": float(best_parent_progress),
                    "best_child_progress": None,
                    "pruned_summary": layer_pruned_totals,
                    "stopped": "no_children",
                }
            )
            break

        deduped = list(_dedupe_nodes(expanded).values())
        deduped.sort(key=lambda node: node.priority, reverse=True)
        best_child_progress = max(node.progress for node in deduped) if deduped else 0.0
        if best_child_progress <= best_parent_progress + float(params.stop_epsilon):
            layer_summaries.append(
                {
                    "depth": int(_depth),
                    "beam_in": beam_in_count,
                    "expanded_children": int(len(expanded)),
                    "deduped_children": int(len(deduped)),
                    "beam_out": 0,
                    "best_parent_progress": float(best_parent_progress),
                    "best_child_progress": float(best_child_progress),
                    "pruned_summary": layer_pruned_totals,
                    "stopped": "stop_epsilon",
                }
            )
            break

        beam = deduped[: params.beam_width]
        layer_summaries.append(
            {
                "depth": int(_depth),
                "beam_in": beam_in_count,
                "expanded_children": int(len(expanded)),
                "deduped_children": int(len(deduped)),
                "beam_out": int(len(beam)),
                "best_parent_progress": float(best_parent_progress),
                "best_child_progress": float(best_child_progress),
                "pruned_summary": layer_pruned_totals,
                "stopped": None,
            }
        )
        for node in beam:
            _store_retained_node(retained, node)
        completed.extend(beam)

    finalized = list(_dedupe_nodes(completed).values())
    finalized.sort(key=_node_sort_key, reverse=True)
    trace = _build_trace_payload(retained=retained, finalized=finalized)
    candidates = [
        _node_to_candidate(
            node=node,
            baseline=baseline,
            search_node_id=trace["state_to_node_id"][node.state_key],
        )
        for node in finalized
    ]
    return candidates, {"root_id": trace["root_id"], "nodes": trace["nodes"], "layer_summaries": layer_summaries}


def generate_target_beam_candidates(
    *,
    baseline_mixture: np.ndarray,
    edges: list[SearchEdge],
    target_context: TargetResidualContext,
    config: TargetBeamSearchConfig | None = None,
) -> list[CandidateAction]:
    candidates, _trace = generate_target_beam_candidates_with_trace(
        baseline_mixture=baseline_mixture,
        edges=edges,
        target_context=target_context,
        config=config,
    )
    return candidates


def generate_target_beam_candidates_with_trace(
    *,
    baseline_mixture: np.ndarray,
    edges: list[SearchEdge],
    target_context: TargetResidualContext,
    config: TargetBeamSearchConfig | None = None,
) -> tuple[list[CandidateAction], dict[str, object]]:
    params = config or TargetBeamSearchConfig()
    baseline = np.asarray(baseline_mixture, dtype=np.float32)
    if baseline.ndim != 1:
        raise ValueError("baseline_mixture must be a 1D array")
    if baseline.shape != np.asarray(target_context.baseline_mixture, dtype=np.float32).shape:
        raise ValueError("baseline_mixture and target_context.baseline_mixture must have the same shape")

    donor_floors = (baseline * float(params.donor_keep_ratio)).astype(np.float32)
    receiver_caps = np.minimum(
        np.ones_like(baseline, dtype=np.float32),
        baseline + float(params.receiver_headroom),
    ).astype(np.float32)

    root = _make_target_node(
        mixture=baseline,
        baseline=baseline,
        plan=[],
        last_edge=None,
        edges=edges,
        target_context=target_context,
        donor_floors=donor_floors,
        receiver_caps=receiver_caps,
        config=params,
        depth=0,
        parent_key=None,
        action=None,
    )
    beam = [root]
    completed: list[_SearchNode] = []
    retained: dict[tuple[int, ...], _SearchNode] = {root.state_key: root}
    layer_summaries: list[dict[str, object]] = []

    for _depth in range(params.max_depth):
        beam_in_count = int(len(beam))
        expanded: list[_SearchNode] = []
        best_parent_progress = max(node.progress for node in beam) if beam else 0.0
        layer_pruned_totals: dict[str, int] = {}
        for node in beam:
            children, pruned_summary = _expand_target_node(
                node=node,
                baseline=baseline,
                edges=edges,
                target_context=target_context,
                donor_floors=donor_floors,
                receiver_caps=receiver_caps,
                config=params,
            )
            node.expanded = True
            node.pruned_summary = pruned_summary
            retained[node.state_key] = node
            expanded.extend(children)
            for reason, count in pruned_summary.items():
                layer_pruned_totals[reason] = layer_pruned_totals.get(reason, 0) + int(count)

        if not expanded:
            layer_summaries.append(
                {
                    "depth": int(_depth),
                    "beam_in": beam_in_count,
                    "expanded_children": 0,
                    "deduped_children": 0,
                    "beam_out": 0,
                    "best_parent_progress": float(best_parent_progress),
                    "best_child_progress": None,
                    "pruned_summary": layer_pruned_totals,
                    "stopped": "no_children",
                }
            )
            break

        deduped = list(_dedupe_nodes(expanded).values())
        deduped.sort(key=lambda node: node.priority, reverse=True)
        best_child_progress = max(node.progress for node in deduped) if deduped else 0.0
        if best_child_progress <= best_parent_progress + float(params.stop_epsilon):
            layer_summaries.append(
                {
                    "depth": int(_depth),
                    "beam_in": beam_in_count,
                    "expanded_children": int(len(expanded)),
                    "deduped_children": int(len(deduped)),
                    "beam_out": 0,
                    "best_parent_progress": float(best_parent_progress),
                    "best_child_progress": float(best_child_progress),
                    "pruned_summary": layer_pruned_totals,
                    "stopped": "stop_epsilon",
                }
            )
            break

        beam = deduped[: params.beam_width]
        layer_summaries.append(
                {
                    "depth": int(_depth),
                    "beam_in": beam_in_count,
                    "expanded_children": int(len(expanded)),
                    "deduped_children": int(len(deduped)),
                    "beam_out": int(len(beam)),
                    "best_parent_progress": float(best_parent_progress),
                    "best_child_progress": float(best_child_progress),
                    "pruned_summary": layer_pruned_totals,
                    "stopped": None,
                }
        )
        for node in beam:
            _store_retained_node(retained, node)
        completed.extend(beam)

    finalized = list(_dedupe_nodes(completed).values())
    finalized.sort(key=_node_sort_key, reverse=True)
    trace = _build_trace_payload(retained=retained, finalized=finalized)
    candidates = [
        _node_to_candidate(
            node=node,
            baseline=baseline,
            search_node_id=trace["state_to_node_id"][node.state_key],
        )
        for node in finalized
    ]
    return candidates, {"root_id": trace["root_id"], "nodes": trace["nodes"], "layer_summaries": layer_summaries}


def _gap(mixture: np.ndarray, pool: np.ndarray) -> float:
    return float(np.abs(np.asarray(mixture, dtype=np.float32) - np.asarray(pool, dtype=np.float32)).sum())


def _dynamic_delta_max(
    *,
    mixture: np.ndarray,
    edge: SearchEdge,
    donor_floors: np.ndarray,
    receiver_caps: np.ndarray,
) -> float:
    donor_headroom = max(0.0, float(mixture[edge.donor] - donor_floors[edge.donor]))
    receiver_headroom = max(0.0, float(receiver_caps[edge.receiver] - mixture[edge.receiver]))
    return float(min(donor_headroom, receiver_headroom, edge.amplitude_band[1]))


def _current_target_gap(*, mixture: np.ndarray, target_context: TargetResidualContext) -> float:
    return compute_target_residual_gap(context=target_context, mixture=mixture)


def _current_portrait_gap(
    *,
    mixture: np.ndarray,
    pool: np.ndarray,
    portrait_context: PortraitResidualContext | None,
) -> float:
    if portrait_context is not None:
        return compute_portrait_residual_gap(context=portrait_context, mixture=mixture)
    return _gap(mixture, pool)


def _resolve_node_min_transfer(
    *,
    delta_max_values: list[float],
    quantile: float,
    floor_min: float,
    cap: float,
) -> float:
    positive = [float(value) for value in delta_max_values if value > AMPLITUDE_DEDUP_EPSILON]
    if not positive:
        return 0.0
    max_value = max(positive)
    raw = float(np.quantile(np.asarray(positive, dtype=np.float32), float(np.clip(quantile, 0.0, 1.0))))
    lower = min(max_value, max(float(floor_min), 0.0))
    node_floor = min(max_value, max(lower, raw))
    if cap > 0.0:
        node_floor = min(node_floor, float(cap))
    return float(min(max_value, max(0.0, node_floor)))


def _best_portrait_amplitude(
    *,
    mixture: np.ndarray,
    pool: np.ndarray,
    edge: SearchEdge,
    portrait_context: PortraitResidualContext | None,
    donor_floors: np.ndarray,
    receiver_caps: np.ndarray,
    min_transfer_mass: float,
) -> tuple[float, float]:
    delta_max = _dynamic_delta_max(
        mixture=mixture,
        edge=edge,
        donor_floors=donor_floors,
        receiver_caps=receiver_caps,
    )
    if delta_max + AMPLITUDE_DEDUP_EPSILON < float(min_transfer_mass):
        return 0.0, float("inf")

    candidates = np.linspace(float(min_transfer_mass), float(delta_max), num=5, dtype=np.float32)
    best_amplitude = float(min_transfer_mass)
    best_gap = float("inf")
    for amplitude in candidates.tolist():
        next_mixture = np.asarray(mixture, dtype=np.float32).copy()
        next_mixture[edge.donor] -= float(amplitude)
        next_mixture[edge.receiver] += float(amplitude)
        if np.any(next_mixture < -1e-8):
            continue
        gap = _current_portrait_gap(mixture=next_mixture, pool=pool, portrait_context=portrait_context)
        if gap < best_gap:
            best_gap = float(gap)
            best_amplitude = float(amplitude)
    return best_amplitude, best_gap


def _best_target_amplitude(
    *,
    mixture: np.ndarray,
    edge: SearchEdge,
    target_context: TargetResidualContext,
    donor_floors: np.ndarray,
    receiver_caps: np.ndarray,
    min_transfer_mass: float,
) -> tuple[float, float]:
    delta_max = _dynamic_delta_max(
        mixture=mixture,
        edge=edge,
        donor_floors=donor_floors,
        receiver_caps=receiver_caps,
    )
    if delta_max + AMPLITUDE_DEDUP_EPSILON < float(min_transfer_mass):
        return 0.0, float("inf")

    candidates = np.linspace(float(min_transfer_mass), float(delta_max), num=5, dtype=np.float32)
    best_amplitude = float(min_transfer_mass)
    best_gap = float("inf")
    for amplitude in candidates.tolist():
        next_mixture = np.asarray(mixture, dtype=np.float32).copy()
        next_mixture[edge.donor] -= float(amplitude)
        next_mixture[edge.receiver] += float(amplitude)
        if np.any(next_mixture < -1e-8):
            continue
        gap = _current_target_gap(mixture=next_mixture, target_context=target_context)
        if gap < best_gap:
            best_gap = float(gap)
            best_amplitude = float(amplitude)
    return best_amplitude, best_gap


def _portrait_opportunity(
    *,
    mixture: np.ndarray,
    pool: np.ndarray,
    portrait_context: PortraitResidualContext | None,
    edges: list[SearchEdge],
    donor_floors: np.ndarray,
    receiver_caps: np.ndarray,
    config: BeamSearchConfig,
    last_edge: tuple[int, int] | None,
) -> float:
    current_gap = _current_portrait_gap(mixture=mixture, pool=pool, portrait_context=portrait_context)
    feasible_edges: list[tuple[SearchEdge, float]] = []
    for edge in edges:
        if last_edge is not None and (edge.donor, edge.receiver) == last_edge:
            continue
        delta_max = _dynamic_delta_max(
            mixture=mixture,
            edge=edge,
            donor_floors=donor_floors,
            receiver_caps=receiver_caps,
        )
        if delta_max > AMPLITUDE_DEDUP_EPSILON:
            feasible_edges.append((edge, delta_max))
    node_min_transfer = _resolve_node_min_transfer(
        delta_max_values=[delta_max for _edge, delta_max in feasible_edges],
        quantile=float(config.adaptive_min_transfer_quantile),
        floor_min=float(config.adaptive_min_transfer_floor_min),
        cap=float(config.min_transfer_mass),
    )
    if node_min_transfer <= 0.0:
        return 0.0

    scores: list[float] = []
    for edge, delta_max in feasible_edges:
        if delta_max + AMPLITUDE_DEDUP_EPSILON < node_min_transfer:
            continue
        _best_amplitude, best_gap = _best_portrait_amplitude(
            mixture=mixture,
            pool=pool,
            edge=edge,
            portrait_context=portrait_context,
            donor_floors=donor_floors,
            receiver_caps=receiver_caps,
            min_transfer_mass=node_min_transfer,
        )
        if np.isfinite(best_gap):
            scores.append(max(0.0, float(current_gap - best_gap)))
    if not scores:
        return 0.0
    top_scores = sorted(scores, reverse=True)[: min(OPPORTUNITY_TOP_K, len(scores))]
    return float(sum(top_scores) / len(top_scores))


def _complexity(
    *,
    mixture: np.ndarray,
    baseline: np.ndarray,
    plan: list[dict[str, float | int]],
    donor_floors: np.ndarray,
    receiver_caps: np.ndarray,
) -> float:
    if not plan:
        return 0.0

    touched = sorted({int(step["donor"]) for step in plan} | {int(step["receiver"]) for step in plan})
    support_size = int(np.count_nonzero(np.abs(mixture - baseline) > 1e-8))
    support_component = 0.0 if mixture.size <= 1 else min(1.0, float(support_size / max(1, mixture.size - 1)))

    boundary_scores: list[float] = []
    for index in touched:
        donor_margin = 1.0
        if baseline[index] > donor_floors[index] + 1e-8:
            donor_margin = max(
                0.0,
                min(1.0, float((mixture[index] - donor_floors[index]) / (baseline[index] - donor_floors[index]))),
            )
        receiver_margin = 1.0
        if receiver_caps[index] > baseline[index] + 1e-8:
            receiver_margin = max(
                0.0,
                min(1.0, float((receiver_caps[index] - mixture[index]) / (receiver_caps[index] - baseline[index]))),
            )
        boundary_scores.append(1.0 - min(donor_margin, receiver_margin))
    boundary_component = float(sum(boundary_scores) / len(boundary_scores)) if boundary_scores else 0.0

    donors = [int(step["donor"]) for step in plan]
    receivers = [int(step["receiver"]) for step in plan]
    repeated = max(0, len(donors) - len(set(donors))) + max(0, len(receivers) - len(set(receivers)))
    redundancy_component = 0.0 if not plan else min(1.0, float(repeated / max(1, 2 * len(plan))))
    return float((support_component + boundary_component + redundancy_component) / 3.0)


def _canonicalize_plan(
    *,
    delta_q: np.ndarray,
    edges: list[SearchEdge],
) -> list[dict[str, float | int]]:
    edge_lookup = {(edge.donor, edge.receiver): float(edge.score) for edge in edges}
    residual = np.asarray(delta_q, dtype=np.float32).copy()
    donors = [(index, float(-value)) for index, value in enumerate(residual) if value < -1e-8]
    receivers = [(index, float(value)) for index, value in enumerate(residual) if value > 1e-8]
    plan: list[dict[str, float | int]] = []
    donor_ptr = 0
    receiver_ptr = 0
    while donor_ptr < len(donors) and receiver_ptr < len(receivers):
        donor, donor_mass = donors[donor_ptr]
        receiver, receiver_mass = receivers[receiver_ptr]
        amplitude = float(min(donor_mass, receiver_mass))
        plan.append(
            {
                "donor": int(donor),
                "receiver": int(receiver),
                "amplitude": amplitude,
                "score": float(edge_lookup.get((donor, receiver), 0.0)),
            }
        )
        donor_mass -= amplitude
        receiver_mass -= amplitude
        if donor_mass <= 1e-8:
            donor_ptr += 1
        else:
            donors[donor_ptr] = (donor, donor_mass)
        if receiver_mass <= 1e-8:
            receiver_ptr += 1
        else:
            receivers[receiver_ptr] = (receiver, receiver_mass)
    return plan


def _state_key(mixture: np.ndarray) -> tuple[int, ...]:
    return tuple(np.round(np.asarray(mixture, dtype=np.float32) / STATE_DEDUP_EPSILON).astype(np.int64).tolist())


def _make_node(
    *,
    mixture: np.ndarray,
    baseline: np.ndarray,
    pool: np.ndarray,
    plan: list[dict[str, float | int]],
    last_edge: tuple[int, int] | None,
    edges: list[SearchEdge],
    baseline_gap: float,
    portrait_context: PortraitResidualContext | None,
    donor_floors: np.ndarray,
    receiver_caps: np.ndarray,
    config: BeamSearchConfig,
    depth: int,
    parent_key: tuple[int, ...] | None,
    action: dict[str, float | int] | None,
) -> _SearchNode:
    delta_q = (np.asarray(mixture, dtype=np.float32) - np.asarray(baseline, dtype=np.float32)).astype(np.float32)
    canonical_plan = _canonicalize_plan(delta_q=delta_q, edges=edges)
    current_gap = _current_portrait_gap(mixture=mixture, pool=pool, portrait_context=portrait_context)
    progress = 0.0 if baseline_gap <= 1e-8 else float((baseline_gap - current_gap) / baseline_gap)
    complexity = _complexity(
        mixture=mixture,
        baseline=baseline,
        plan=canonical_plan,
        donor_floors=donor_floors,
        receiver_caps=receiver_caps,
    )
    effective_progress = float(progress / (1.0 + complexity))
    opportunity = _portrait_opportunity(
        mixture=mixture,
        pool=pool,
        portrait_context=portrait_context,
        edges=edges,
        donor_floors=donor_floors,
        receiver_caps=receiver_caps,
        config=config,
        last_edge=last_edge,
    )
    priority = float(effective_progress + float(config.lambda_opportunity) * opportunity)
    return _SearchNode(
        state_key=_state_key(mixture),
        parent_key=parent_key,
        depth=depth,
        action=action,
        mixture=np.asarray(mixture, dtype=np.float32),
        delta_q=delta_q,
        plan=canonical_plan,
        priority=priority,
        progress=progress,
        opportunity=opportunity,
        complexity=complexity,
        last_edge=last_edge,
    )


def _target_opportunity(
    *,
    mixture: np.ndarray,
    edges: list[SearchEdge],
    target_context: TargetResidualContext,
    donor_floors: np.ndarray,
    receiver_caps: np.ndarray,
    config: TargetBeamSearchConfig,
    last_edge: tuple[int, int] | None,
) -> float:
    current_gap = _current_target_gap(mixture=mixture, target_context=target_context)
    feasible_edges: list[tuple[SearchEdge, float]] = []
    for edge in edges:
        if last_edge is not None and (edge.donor, edge.receiver) == last_edge:
            continue
        delta_max = _dynamic_delta_max(
            mixture=mixture,
            edge=edge,
            donor_floors=donor_floors,
            receiver_caps=receiver_caps,
        )
        if delta_max > AMPLITUDE_DEDUP_EPSILON:
            feasible_edges.append((edge, delta_max))
    node_min_transfer = _resolve_node_min_transfer(
        delta_max_values=[delta_max for _edge, delta_max in feasible_edges],
        quantile=float(config.adaptive_min_transfer_quantile),
        floor_min=float(config.adaptive_min_transfer_floor_min),
        cap=float(config.min_transfer_mass),
    )
    if node_min_transfer <= 0.0:
        return 0.0

    scores: list[float] = []
    for edge, delta_max in feasible_edges:
        if delta_max + AMPLITUDE_DEDUP_EPSILON < node_min_transfer:
            continue
        best_amplitude, best_gap = _best_target_amplitude(
            mixture=mixture,
            edge=edge,
            target_context=target_context,
            donor_floors=donor_floors,
            receiver_caps=receiver_caps,
            min_transfer_mass=node_min_transfer,
        )
        if best_amplitude <= 0.0 or not np.isfinite(best_gap):
            continue
        improvement = max(0.0, float(current_gap - best_gap))
        scores.append(float(improvement))
    if not scores:
        return 0.0
    top_scores = sorted(scores, reverse=True)[: min(OPPORTUNITY_TOP_K, len(scores))]
    return float(sum(top_scores) / len(top_scores))


def _make_target_node(
    *,
    mixture: np.ndarray,
    baseline: np.ndarray,
    plan: list[dict[str, float | int]],
    last_edge: tuple[int, int] | None,
    edges: list[SearchEdge],
    target_context: TargetResidualContext,
    donor_floors: np.ndarray,
    receiver_caps: np.ndarray,
    config: TargetBeamSearchConfig,
    depth: int,
    parent_key: tuple[int, ...] | None,
    action: dict[str, float | int] | None,
) -> _SearchNode:
    delta_q = (np.asarray(mixture, dtype=np.float32) - np.asarray(baseline, dtype=np.float32)).astype(np.float32)
    canonical_plan = _canonicalize_plan(delta_q=delta_q, edges=edges)
    current_gap = compute_target_residual_gap(context=target_context, mixture=mixture)
    baseline_gap = float(target_context.baseline_gap)
    if np.allclose(np.asarray(mixture, dtype=np.float32), np.asarray(target_context.baseline_mixture, dtype=np.float32), atol=1e-8):
        progress = 0.0
    else:
        progress = 0.0 if baseline_gap <= 1e-8 else float((baseline_gap - current_gap) / baseline_gap)
    complexity = _complexity(
        mixture=mixture,
        baseline=baseline,
        plan=canonical_plan,
        donor_floors=donor_floors,
        receiver_caps=receiver_caps,
    )
    effective_progress = float(progress / (1.0 + complexity))
    opportunity = _target_opportunity(
        mixture=mixture,
        edges=edges,
        target_context=target_context,
        donor_floors=donor_floors,
        receiver_caps=receiver_caps,
        config=config,
        last_edge=last_edge,
    )
    priority = float(effective_progress + float(config.lambda_opportunity) * opportunity)
    return _SearchNode(
        state_key=_state_key(mixture),
        parent_key=parent_key,
        depth=depth,
        action=action,
        mixture=np.asarray(mixture, dtype=np.float32),
        delta_q=delta_q,
        plan=canonical_plan,
        priority=priority,
        progress=progress,
        opportunity=opportunity,
        complexity=complexity,
        last_edge=last_edge,
    )


def _merge_plan(
    plan: list[dict[str, float | int]],
    *,
    donor: int,
    receiver: int,
    amplitude: float,
    edge_score: float,
) -> list[dict[str, float | int]]:
    merged = [dict(step) for step in plan]
    for step in merged:
        if int(step["donor"]) == donor and int(step["receiver"]) == receiver:
            step["amplitude"] = float(step["amplitude"]) + float(amplitude)
            step["score"] = max(float(step.get("score", 0.0)), float(edge_score))
            return merged
    merged.append(
        {
            "donor": int(donor),
            "receiver": int(receiver),
            "amplitude": float(amplitude),
            "score": float(edge_score),
        }
    )
    return merged


def _expand_node(
    *,
    node: _SearchNode,
    baseline: np.ndarray,
    pool: np.ndarray,
    edges: list[SearchEdge],
    baseline_gap: float,
    portrait_context: PortraitResidualContext | None,
    donor_floors: np.ndarray,
    receiver_caps: np.ndarray,
    config: BeamSearchConfig,
) -> tuple[list[_SearchNode], dict[str, int]]:
    scored_edges: list[tuple[float, float, SearchEdge, float, float]] = []
    pruned_summary = {
        "repeat_edge": 0,
        "infeasible": 0,
        "proposal_pruned": 0,
        "no_gain": 0,
    }
    current_gap = _current_portrait_gap(mixture=node.mixture, pool=pool, portrait_context=portrait_context)
    feasible_edges: list[tuple[SearchEdge, float]] = []
    for edge in edges:
        if node.last_edge is not None and (edge.donor, edge.receiver) == node.last_edge:
            pruned_summary["repeat_edge"] += 1
            continue
        delta_max = _dynamic_delta_max(
            mixture=node.mixture,
            edge=edge,
            donor_floors=donor_floors,
            receiver_caps=receiver_caps,
        )
        if delta_max <= AMPLITUDE_DEDUP_EPSILON:
            pruned_summary["infeasible"] += 1
            continue
        feasible_edges.append((edge, delta_max))

    node_min_transfer = _resolve_node_min_transfer(
        delta_max_values=[delta_max for _edge, delta_max in feasible_edges],
        quantile=float(config.adaptive_min_transfer_quantile),
        floor_min=float(config.adaptive_min_transfer_floor_min),
        cap=float(config.min_transfer_mass),
    )
    if node_min_transfer <= 0.0:
        return [], pruned_summary

    for edge, delta_max in feasible_edges:
        if delta_max + AMPLITUDE_DEDUP_EPSILON < node_min_transfer:
            pruned_summary["infeasible"] += 1
            continue
        best_amplitude, best_gap = _best_portrait_amplitude(
            mixture=node.mixture,
            pool=pool,
            edge=edge,
            portrait_context=portrait_context,
            donor_floors=donor_floors,
            receiver_caps=receiver_caps,
            min_transfer_mass=node_min_transfer,
        )
        if best_amplitude <= 0.0 or not np.isfinite(best_gap):
            pruned_summary["infeasible"] += 1
            continue
        improvement = max(0.0, float(current_gap - best_gap))
        scored_edges.append((improvement, float(edge.score), edge, best_amplitude, delta_max))

    scored_edges.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if len(scored_edges) > config.proposal_edges_per_node:
        pruned_summary["proposal_pruned"] += len(scored_edges) - config.proposal_edges_per_node
    children: list[_SearchNode] = []
    for improvement, _edge_score, edge, best_amplitude, delta_max in scored_edges[: config.proposal_edges_per_node]:
        if improvement <= 0.0:
            pruned_summary["infeasible"] += 1
            continue
        amplitudes = get_depth_adaptive_amplitudes(
            delta_max=delta_max,
            min_transfer_mass=node_min_transfer,
            best_amplitude=float(best_amplitude),
            depth=node.depth,
        )
        for amplitude in amplitudes:
            next_mixture = node.mixture.copy()
            next_mixture[edge.donor] -= float(amplitude)
            next_mixture[edge.receiver] += float(amplitude)
            if np.any(next_mixture < -1e-8):
                continue
            next_plan = _merge_plan(
                node.plan,
                donor=edge.donor,
                receiver=edge.receiver,
                amplitude=float(amplitude),
                edge_score=edge.score,
            )
            child = _make_node(
                mixture=next_mixture,
                baseline=baseline,
                pool=pool,
                plan=next_plan,
                last_edge=(edge.donor, edge.receiver),
                edges=edges,
                baseline_gap=baseline_gap,
                portrait_context=portrait_context,
                donor_floors=donor_floors,
                receiver_caps=receiver_caps,
                config=config,
                depth=node.depth + 1,
                parent_key=node.state_key,
                action={
                    "donor": int(edge.donor),
                    "receiver": int(edge.receiver),
                    "amplitude": float(amplitude),
                },
            )
            if child.progress > node.progress + float(config.stop_epsilon):
                children.append(child)
            else:
                pruned_summary["no_gain"] += 1
    return children, pruned_summary


def _expand_target_node(
    *,
    node: _SearchNode,
    baseline: np.ndarray,
    edges: list[SearchEdge],
    target_context: TargetResidualContext,
    donor_floors: np.ndarray,
    receiver_caps: np.ndarray,
    config: TargetBeamSearchConfig,
) -> tuple[list[_SearchNode], dict[str, int]]:
    scored_edges: list[tuple[float, float, SearchEdge, float, float]] = []
    pruned_summary = {
        "repeat_edge": 0,
        "infeasible": 0,
        "proposal_pruned": 0,
        "no_gain": 0,
    }
    current_gap = _current_target_gap(mixture=node.mixture, target_context=target_context)
    feasible_edges: list[tuple[SearchEdge, float]] = []
    for edge in edges:
        if node.last_edge is not None and (edge.donor, edge.receiver) == node.last_edge:
            pruned_summary["repeat_edge"] += 1
            continue
        delta_max = _dynamic_delta_max(
            mixture=node.mixture,
            edge=edge,
            donor_floors=donor_floors,
            receiver_caps=receiver_caps,
        )
        if delta_max <= AMPLITUDE_DEDUP_EPSILON:
            pruned_summary["infeasible"] += 1
            continue
        feasible_edges.append((edge, delta_max))

    node_min_transfer = _resolve_node_min_transfer(
        delta_max_values=[delta_max for _edge, delta_max in feasible_edges],
        quantile=float(config.adaptive_min_transfer_quantile),
        floor_min=float(config.adaptive_min_transfer_floor_min),
        cap=float(config.min_transfer_mass),
    )
    if node_min_transfer <= 0.0:
        return [], pruned_summary

    for edge, delta_max in feasible_edges:
        if delta_max + AMPLITUDE_DEDUP_EPSILON < node_min_transfer:
            pruned_summary["infeasible"] += 1
            continue
        best_amplitude, best_gap = _best_target_amplitude(
            mixture=node.mixture,
            edge=edge,
            target_context=target_context,
            donor_floors=donor_floors,
            receiver_caps=receiver_caps,
            min_transfer_mass=node_min_transfer,
        )
        if best_amplitude <= 0.0 or not np.isfinite(best_gap):
            pruned_summary["infeasible"] += 1
            continue
        improvement = max(0.0, float(current_gap - best_gap))
        scored_edges.append((improvement, float(edge.score), edge, best_amplitude, delta_max))

    scored_edges.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if len(scored_edges) > config.proposal_edges_per_node:
        pruned_summary["proposal_pruned"] += len(scored_edges) - config.proposal_edges_per_node

    children: list[_SearchNode] = []
    for improvement, _edge_score, edge, best_amplitude, delta_max in scored_edges[: config.proposal_edges_per_node]:
        if improvement <= 0.0:
            pruned_summary["infeasible"] += 1
            continue
        amplitudes = get_depth_adaptive_amplitudes(
            delta_max=delta_max,
            min_transfer_mass=node_min_transfer,
            best_amplitude=float(best_amplitude),
            depth=node.depth,
        )
        for amplitude in amplitudes:
            next_mixture = node.mixture.copy()
            next_mixture[edge.donor] -= float(amplitude)
            next_mixture[edge.receiver] += float(amplitude)
            if np.any(next_mixture < -1e-8):
                continue
            next_plan = _merge_plan(
                node.plan,
                donor=edge.donor,
                receiver=edge.receiver,
                amplitude=float(amplitude),
                edge_score=edge.score,
            )
            child = _make_target_node(
                mixture=next_mixture,
                baseline=baseline,
                plan=next_plan,
                last_edge=(edge.donor, edge.receiver),
                edges=edges,
                target_context=target_context,
                donor_floors=donor_floors,
                receiver_caps=receiver_caps,
                config=config,
                depth=node.depth + 1,
                parent_key=node.state_key,
                action={
                    "donor": int(edge.donor),
                    "receiver": int(edge.receiver),
                    "amplitude": float(amplitude),
                },
            )
            if child.progress > node.progress + float(config.stop_epsilon):
                children.append(child)
            else:
                pruned_summary["no_gain"] += 1
    return children, pruned_summary


def _dedupe_nodes(nodes: list[_SearchNode]) -> dict[tuple[int, ...], _SearchNode]:
    best_by_key: dict[tuple[int, ...], _SearchNode] = {}
    for node in nodes:
        current = best_by_key.get(node.state_key)
        if current is None or _node_sort_key(node) > _node_sort_key(current):
            best_by_key[node.state_key] = node
    return best_by_key


def _node_sort_key(node: _SearchNode) -> tuple[float, float, float, float]:
    return (
        float(node.priority),
        float(node.progress),
        float(-len(node.plan)),
        float(-node.complexity),
    )


def _store_retained_node(retained: dict[tuple[int, ...], _SearchNode], node: _SearchNode) -> None:
    current = retained.get(node.state_key)
    if current is None or _node_sort_key(node) > _node_sort_key(current):
        retained[node.state_key] = node


def _build_trace_payload(
    *,
    retained: dict[tuple[int, ...], _SearchNode],
    finalized: list[_SearchNode],
) -> dict[str, object]:
    nodes = list(retained.values())
    nodes.sort(key=lambda node: (node.depth, -node.priority, node.state_key))
    state_to_node_id = {
        node.state_key: ("root" if node.depth == 0 else f"node_{index}")
        for index, node in enumerate(nodes)
    }
    finalized_keys = {node.state_key for node in finalized}
    exported_nodes: list[dict[str, object]] = []
    for node in nodes:
        node_id = state_to_node_id[node.state_key]
        parent_id = state_to_node_id.get(node.parent_key) if node.parent_key is not None else None
        node_type = "root" if node.depth == 0 else ("partial" if node.expanded else "completed")
        status = "expanded" if node.expanded else ("completed" if node.state_key in finalized_keys else "frontier")
        exported_nodes.append(
            {
                "node_id": node_id,
                "parent_id": parent_id,
                "depth": int(node.depth),
                "node_type": node_type,
                "status": status,
                "action": dict(node.action) if node.action is not None else None,
                "plan": [dict(step) for step in node.plan],
                "delta_q": node.delta_q.astype(np.float32).tolist(),
                "progress": float(node.progress),
                "opportunity": float(node.opportunity),
                "complexity": float(node.complexity),
                "priority": float(node.priority),
                "candidate_id": None,
                "is_recommended": False,
                "pruned_summary": {key: int(value) for key, value in node.pruned_summary.items() if int(value) > 0},
            }
        )
    return {
        "root_id": state_to_node_id[nodes[0].state_key],
        "nodes": exported_nodes,
        "state_to_node_id": state_to_node_id,
    }


def _node_to_candidate(*, node: _SearchNode, baseline: np.ndarray, search_node_id: str) -> CandidateAction:
    support_size = int(np.count_nonzero(np.abs(node.delta_q) > 1e-8))
    donors = sorted({int(step["donor"]) for step in node.plan})
    receivers = sorted({int(step["receiver"]) for step in node.plan})
    total_amplitude = float(sum(float(step["amplitude"]) for step in node.plan))
    return CandidateAction(
        baseline_mixture=np.asarray(baseline, dtype=np.float32).tolist(),
        target_mixture=node.mixture.astype(np.float32).tolist(),
        delta_q=node.delta_q.astype(np.float32).tolist(),
        donors=donors,
        receivers=receivers,
        amplitude=total_amplitude,
        support_size=support_size,
        metadata={
            "plan": node.plan,
            "search_priority": float(node.priority),
            "progress": float(node.progress),
            "opportunity": float(node.opportunity),
            "complexity": float(node.complexity),
            "search_node_id": search_node_id,
        },
    )
