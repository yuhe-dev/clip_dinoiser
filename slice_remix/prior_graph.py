from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np

from .baseline import estimate_baseline_mixture


SHAPE_PREFIXES = ("hist[", "profile[", "delta_profile[")
DIMENSIONS = ("quality", "difficulty", "coverage")
FIXED_SALIENCY_GAMMA = 0.5


@dataclass(frozen=True)
class PriorGraphHyperparams:
    lambda_balance: float = 0.35
    lambda_user: float = 0.25
    lambda_risk: float = 0.25
    shape_rho: float = 0.8
    donor_keep_ratio: float = 0.2
    min_transfer_mass: float = 0.03
    receiver_headroom: float = 0.15
    epsilon: float = 1e-8
    top_k_render: int = 12
    score_threshold: float = 0.0


@dataclass
class PriorGraphUserIntent:
    dimension_weights: dict[str, float] = field(default_factory=dict)
    atomic_block_weights: dict[str, float] = field(default_factory=dict)
    protected_dimensions: set[str] = field(default_factory=set)
    protected_atomic_blocks: set[str] = field(default_factory=set)
    frozen_slices: set[str] = field(default_factory=set)
    stable_slices: set[str] = field(default_factory=set)
    increase_slices: set[str] = field(default_factory=set)
    decrease_slices: set[str] = field(default_factory=set)
    preferred_edges: set[tuple[str, str]] = field(default_factory=set)
    avoided_edges: set[tuple[str, str]] = field(default_factory=set)
    forbidden_edges: set[tuple[str, str]] = field(default_factory=set)


@dataclass(frozen=True)
class PriorGraphNode:
    slice_id: str
    index: int
    canonical_weight: float
    baseline_weight: float
    pool_delta: float
    instability_score: float
    default_action_state: str


@dataclass(frozen=True)
class PriorGraphEdge:
    edge_id: str
    donor: str
    receiver: str
    admissible: bool
    masked_reason: str | None
    score: float
    fit_score: float
    balance_score: float
    user_score: float
    risk_score: float
    block_scores: dict[str, float]
    amplitude_band: tuple[float, float]
    visible_by_default: bool
    bias_score: float = 0.0
    risk_components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PriorGraphPayload:
    nodes: list[PriorGraphNode]
    edges: list[PriorGraphEdge]
    graph_context: dict[str, object]
    defaults: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _BlockLayout:
    name: str
    dimension: str
    labels: tuple[str, ...]
    shape_indices: np.ndarray
    scalar_indices: np.ndarray


@dataclass(frozen=True)
class _BranchDelta:
    delta: np.ndarray
    weights: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class _BlockContexts:
    layout: _BlockLayout
    shape_delta: _BranchDelta
    scalar_delta: _BranchDelta
    slice_shape: np.ndarray
    slice_scalar: np.ndarray
    pool_shape: np.ndarray
    pool_scalar: np.ndarray
    rho: float
    weight: float


@dataclass(frozen=True)
class PortraitResidualContext:
    baseline_mixture: np.ndarray
    pool_mixture: np.ndarray
    block_contexts: tuple[_BlockContexts, ...]
    tau_gap: float
    baseline_gap: float
    epsilon: float


@dataclass
class SearchConstraints:
    protected_dimensions: set[str] = field(default_factory=set)
    protected_blocks: set[str] = field(default_factory=set)
    frozen_slices: set[str] = field(default_factory=set)
    forbidden_edges: set[tuple[str, str]] = field(default_factory=set)


@dataclass
class SearchBias:
    increase_slices: set[str] = field(default_factory=set)
    decrease_slices: set[str] = field(default_factory=set)
    stable_slices: set[str] = field(default_factory=set)
    preferred_edges: set[tuple[str, str]] = field(default_factory=set)
    avoided_edges: set[tuple[str, str]] = field(default_factory=set)


@dataclass(frozen=True)
class SearchStyle:
    name: str = "balanced"
    proposal_scale: float = 1.0
    refinement_bias: int = 0
    stop_epsilon_scale: float = 1.0


@dataclass(frozen=True)
class TargetPortraitSpec:
    shape_targets: dict[str, np.ndarray]
    scalar_targets: dict[str, np.ndarray] = field(default_factory=dict)
    block_weights: dict[str, float] = field(default_factory=dict)
    source: str = "pool_initialized"


@dataclass(frozen=True)
class _TargetBlockContext:
    layout: _BlockLayout
    weight: float
    target_shape: np.ndarray
    target_scalar: np.ndarray
    baseline_shape: np.ndarray
    baseline_scalar: np.ndarray
    slice_shape: np.ndarray
    slice_scalar: np.ndarray
    shape_delta: _BranchDelta
    scalar_delta: _BranchDelta
    rho: float
    support_risk_by_slice: np.ndarray
    empty_flag_by_slice: np.ndarray


@dataclass(frozen=True)
class TargetResidualContext:
    baseline_mixture: np.ndarray
    baseline_gap: float
    target_by_block: dict[str, np.ndarray]
    block_weights: dict[str, float]
    slice_shape_by_block: dict[str, np.ndarray]
    scalar_target_by_block: dict[str, np.ndarray] = field(default_factory=dict)
    slice_scalar_by_block: dict[str, np.ndarray] = field(default_factory=dict)
    block_contexts: tuple[_TargetBlockContext, ...] = ()
    tau_gap: float = 0.0
    epsilon: float = 1e-8


def _safe_array(values: np.ndarray | Iterable[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _normalize_weights(raw: np.ndarray, epsilon: float) -> np.ndarray:
    if raw.size == 0:
        return raw
    total = float(raw.sum())
    if total <= epsilon:
        return np.full(raw.shape, 1.0 / float(raw.size), dtype=np.float32)
    return (raw / total).astype(np.float32)


def _weighted_average(matrix: np.ndarray, weights: np.ndarray, epsilon: float) -> np.ndarray:
    sample_matrix = np.asarray(matrix, dtype=np.float32)
    if sample_matrix.ndim == 1:
        sample_matrix = sample_matrix[:, None]
    vector = np.asarray(weights, dtype=np.float64).reshape(-1)
    if sample_matrix.shape[0] != vector.shape[0]:
        raise ValueError("weight length must match matrix row count")
    total = float(vector.sum())
    if total <= epsilon:
        return np.zeros(sample_matrix.shape[1], dtype=np.float32)
    matrix64 = sample_matrix.astype(np.float64, copy=False)
    return ((vector[:, None] * matrix64).sum(axis=0) / total).astype(np.float32)


def _weighted_slice_average(matrix: np.ndarray, memberships: np.ndarray, epsilon: float) -> np.ndarray:
    sample_matrix = np.asarray(matrix, dtype=np.float32)
    if sample_matrix.ndim == 1:
        sample_matrix = sample_matrix[:, None]
    membership_matrix = np.asarray(memberships, dtype=np.float64)
    safe = np.clip(membership_matrix.sum(axis=0, dtype=np.float64), epsilon, None)
    matrix64 = sample_matrix.astype(np.float64, copy=False)
    weighted_sum = np.einsum("nk,nd->kd", membership_matrix, matrix64, dtype=np.float64)
    return (weighted_sum / safe[:, None]).astype(np.float32)


def _build_block_layouts(feature_groups: dict[str, np.ndarray], feature_label_map: dict[str, list[str]]) -> dict[str, _BlockLayout]:
    layouts: dict[str, _BlockLayout] = {}
    for block_name, matrix in feature_groups.items():
        width = int(np.asarray(matrix).shape[1]) if np.asarray(matrix).ndim == 2 else 1
        labels = feature_label_map.get(block_name, [f"dim_{index}" for index in range(width)])
        shape_indices = [index for index, label in enumerate(labels) if str(label).startswith(SHAPE_PREFIXES)]
        scalar_indices = [index for index, _label in enumerate(labels) if index not in shape_indices]
        dimension = block_name.split(".", 1)[0] if "." in block_name else "quality"
        layouts[block_name] = _BlockLayout(
            name=block_name,
            dimension=dimension,
            labels=tuple(str(label) for label in labels),
            shape_indices=np.asarray(shape_indices, dtype=np.int64),
            scalar_indices=np.asarray(scalar_indices, dtype=np.int64),
        )
    return layouts


def _scalar_label_positions(layout: _BlockLayout) -> dict[str, int]:
    positions: dict[str, int] = {}
    for scalar_position, source_index in enumerate(layout.scalar_indices.tolist()):
        positions[str(layout.labels[source_index])] = scalar_position
    return positions


def _normalize_shape_target(values: np.ndarray, layout: _BlockLayout, epsilon: float) -> np.ndarray:
    target = np.asarray(values, dtype=np.float32).reshape(-1)
    if target.size == 0:
        return target
    shape_labels = [layout.labels[index] for index in layout.shape_indices.tolist()]
    if shape_labels and all(str(label).startswith("hist[") for label in shape_labels):
        total = float(target.sum())
        if total > epsilon:
            return (target / total).astype(np.float32)
    return target.astype(np.float32)


def _split_matrix(matrix: np.ndarray, layout: _BlockLayout) -> tuple[np.ndarray, np.ndarray]:
    sample_matrix = np.asarray(matrix, dtype=np.float32)
    if sample_matrix.ndim == 1:
        sample_matrix = sample_matrix[:, None]
    shape = (
        sample_matrix[:, layout.shape_indices]
        if layout.shape_indices.size
        else np.zeros((sample_matrix.shape[0], 0), dtype=np.float32)
    )
    scalar = (
        sample_matrix[:, layout.scalar_indices]
        if layout.scalar_indices.size
        else np.zeros((sample_matrix.shape[0], 0), dtype=np.float32)
    )
    return shape.astype(np.float32), scalar.astype(np.float32)


def _cumulative_rows(matrix: np.ndarray) -> np.ndarray:
    sample_matrix = np.asarray(matrix, dtype=np.float32)
    if sample_matrix.size == 0:
        return sample_matrix.reshape(sample_matrix.shape[0], 0) if sample_matrix.ndim == 2 else np.zeros((0, 0), dtype=np.float32)
    if sample_matrix.ndim == 1:
        sample_matrix = sample_matrix[None, :]
    return np.cumsum(sample_matrix, axis=1, dtype=np.float32)


def _robust_scale(matrix: np.ndarray, epsilon: float) -> np.ndarray:
    sample_matrix = np.asarray(matrix, dtype=np.float32)
    if sample_matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if sample_matrix.ndim == 1:
        sample_matrix = sample_matrix[:, None]
    median = np.median(sample_matrix, axis=0)
    mad = np.median(np.abs(sample_matrix - median), axis=0)
    scale = 1.4826 * mad
    scale = np.where(scale > epsilon, scale, 1.0).astype(np.float32)
    return scale


def _standardize(vector: np.ndarray, scale: np.ndarray, epsilon: float) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    safe_scale = np.asarray(scale, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    safe_scale = np.where(safe_scale > epsilon, safe_scale, 1.0)
    return (values / safe_scale).astype(np.float32)


def _soft_saliency(delta: np.ndarray, gamma: float, epsilon: float) -> np.ndarray:
    values = np.asarray(delta, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    raw = np.power(np.abs(values) + epsilon, gamma, dtype=np.float32)
    return _normalize_weights(raw, epsilon)


def _weighted_norm(values: np.ndarray, weights: np.ndarray, epsilon: float) -> float:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        return 0.0
    weight_vector = np.asarray(weights, dtype=np.float32).reshape(-1)
    return float(np.sqrt(np.sum(weight_vector * (vector**2), dtype=np.float32) + epsilon))


def _weighted_cosine(x: np.ndarray, y: np.ndarray, weights: np.ndarray, epsilon: float) -> float:
    x_vec = np.asarray(x, dtype=np.float32).reshape(-1)
    y_vec = np.asarray(y, dtype=np.float32).reshape(-1)
    if x_vec.size == 0 or y_vec.size == 0:
        return 0.0
    weight_vec = np.asarray(weights, dtype=np.float32).reshape(-1)
    numerator = float(np.sum(weight_vec * x_vec * y_vec, dtype=np.float32))
    denominator = _weighted_norm(x_vec, weight_vec, epsilon) * _weighted_norm(y_vec, weight_vec, epsilon)
    if denominator <= epsilon:
        return 0.0
    return float(np.clip(numerator / denominator, -1.0, 1.0))


def _gap_suppression(delta: np.ndarray, weights: np.ndarray, tau_gap: float, epsilon: float) -> float:
    norm = _weighted_norm(delta, weights, epsilon)
    return float(norm / (norm + tau_gap + epsilon))


def _amplitude_factor(effect: np.ndarray, delta: np.ndarray, weights: np.ndarray, epsilon: float) -> float:
    delta_norm = _weighted_norm(delta, weights, epsilon)
    effect_norm = _weighted_norm(effect, weights, epsilon)
    return float(min(1.0, effect_norm / (delta_norm + epsilon)))


def _branch_score(delta: np.ndarray, effect: np.ndarray, weights: np.ndarray, tau_gap: float, epsilon: float) -> float:
    if delta.size == 0 or effect.size == 0:
        return 0.0
    return (
        _gap_suppression(delta, weights, tau_gap, epsilon)
        * _weighted_cosine(effect, delta, weights, epsilon)
        * _amplitude_factor(effect, delta, weights, epsilon)
    )


def _dimension_balanced_defaults(block_names: Iterable[str]) -> dict[str, float]:
    groups: dict[str, list[str]] = {dimension: [] for dimension in DIMENSIONS}
    for block_name in block_names:
        dimension = block_name.split(".", 1)[0] if "." in block_name else "quality"
        groups.setdefault(dimension, []).append(block_name)

    weights: dict[str, float] = {}
    for dimension, names in groups.items():
        if not names:
            continue
        total_dimension_weight = 1.0 / float(len([group for group in groups.values() if group]))
        per_block = total_dimension_weight / float(len(names))
        for name in names:
            weights[name] = float(per_block)
    return weights


def _resolve_user_block_weight(
    *,
    block_name: str,
    layout: _BlockLayout,
    default_weight: float,
    user_intent: PriorGraphUserIntent,
) -> float:
    dimension_weight = float(user_intent.dimension_weights.get(layout.dimension, 1.0))
    block_weight = float(user_intent.atomic_block_weights.get(block_name, 1.0))
    return default_weight * dimension_weight * block_weight


def _resolve_block_rho(*, has_shape: bool, has_scalar: bool, shape_rho: float) -> float:
    if has_shape and has_scalar:
        return float(shape_rho)
    if has_shape:
        return 1.0
    if has_scalar:
        return 0.0
    return 1.0


def _compute_slice_instability(memberships: np.ndarray, epsilon: float) -> np.ndarray:
    membership_matrix = np.asarray(memberships, dtype=np.float32)
    num_slices = int(membership_matrix.shape[1])
    if num_slices <= 1:
        return np.zeros((num_slices,), dtype=np.float32)

    entropy = -np.sum(membership_matrix * np.log(np.clip(membership_matrix, epsilon, None)), axis=1, dtype=np.float32)
    entropy /= np.log(float(num_slices))
    slice_scores = []
    for slice_index in range(num_slices):
        slice_scores.append(_weighted_average(entropy, membership_matrix[:, slice_index], epsilon)[0])
    return np.asarray(slice_scores, dtype=np.float32)


def _default_action_state(pool_delta: float, tolerance: float) -> str:
    if pool_delta > tolerance:
        return "increase"
    if pool_delta < -tolerance:
        return "decrease"
    return "inspect"


def _mask_reason(
    *,
    donor: str,
    receiver: str,
    donor_index: int,
    receiver_index: int,
    baseline_mixture: np.ndarray,
    pool_mixture: np.ndarray,
    user_intent: PriorGraphUserIntent,
    hyperparams: PriorGraphHyperparams,
) -> tuple[bool, str | None, float, float]:
    if donor == receiver:
        return False, "self_edge", 0.0, 0.0
    if donor in user_intent.frozen_slices or receiver in user_intent.frozen_slices:
        return False, "frozen_slice", 0.0, 0.0
    if (donor, receiver) in user_intent.forbidden_edges:
        return False, "forbidden_edge", 0.0, 0.0

    donor_mass = float(baseline_mixture[donor_index])
    donor_transferable = max(
        0.0,
        donor_mass - max(hyperparams.min_transfer_mass, hyperparams.donor_keep_ratio * donor_mass),
    )
    receiver_cap = min(1.0, float(pool_mixture[receiver_index]) + hyperparams.receiver_headroom)
    receiver_slack = max(0.0, receiver_cap - float(baseline_mixture[receiver_index]))

    if donor_transferable <= hyperparams.epsilon:
        return False, "donor_too_small", donor_transferable, receiver_slack
    if receiver_slack <= hyperparams.epsilon:
        return False, "receiver_at_cap", donor_transferable, receiver_slack
    return True, None, donor_transferable, receiver_slack


def _user_score(donor: str, receiver: str, user_intent: PriorGraphUserIntent) -> float:
    score = 0.0
    if (donor, receiver) in user_intent.preferred_edges:
        score += 1.0
    if (donor, receiver) in user_intent.avoided_edges:
        score -= 1.0
    if receiver in user_intent.increase_slices:
        score += 0.5
    if donor in user_intent.increase_slices:
        score -= 0.5
    if donor in user_intent.decrease_slices:
        score += 0.5
    if receiver in user_intent.decrease_slices:
        score -= 0.5
    if donor in user_intent.stable_slices:
        score -= 0.5
    if receiver in user_intent.stable_slices:
        score -= 0.5
    return float(score)


def _protected_blocks(user_intent: PriorGraphUserIntent, block_name: str, layout: _BlockLayout) -> bool:
    return block_name in user_intent.protected_atomic_blocks or layout.dimension in user_intent.protected_dimensions


def _compute_portrait_residual_gap_from_contexts(
    *,
    mixture: np.ndarray,
    block_contexts: Iterable[_BlockContexts],
    epsilon: float,
) -> float:
    weights = np.asarray(mixture, dtype=np.float32).reshape(-1)
    weighted_gap_sum = 0.0
    weight_sum = 0.0
    for context in block_contexts:
        expected_shape = weights @ context.slice_shape if context.slice_shape.size else np.zeros((0,), dtype=np.float32)
        expected_scalar = weights @ context.slice_scalar if context.slice_scalar.size else np.zeros((0,), dtype=np.float32)
        shape_residual = _standardize(context.pool_shape - expected_shape, context.shape_delta.scale, epsilon)
        scalar_residual = _standardize(context.pool_scalar - expected_scalar, context.scalar_delta.scale, epsilon)
        shape_gap = _weighted_norm(shape_residual, context.shape_delta.weights, epsilon)
        scalar_gap = _weighted_norm(scalar_residual, context.scalar_delta.weights, epsilon)
        block_gap = float(context.rho * shape_gap + (1.0 - context.rho) * scalar_gap)
        weighted_gap_sum += context.weight * block_gap
        weight_sum += context.weight
    return float(weighted_gap_sum / (weight_sum + epsilon))


def compute_portrait_residual_gap(
    *,
    context: PortraitResidualContext,
    mixture: np.ndarray,
) -> float:
    return _compute_portrait_residual_gap_from_contexts(
        mixture=mixture,
        block_contexts=context.block_contexts,
        epsilon=float(context.epsilon),
    )


def build_portrait_residual_context(
    *,
    feature_groups: dict[str, np.ndarray],
    feature_label_map: dict[str, list[str]],
    memberships: np.ndarray,
    baseline_sample_indices: list[int],
    hyperparams: PriorGraphHyperparams | None = None,
    user_intent: PriorGraphUserIntent | None = None,
) -> PortraitResidualContext:
    membership_matrix = np.asarray(memberships, dtype=np.float32)
    if membership_matrix.ndim != 2:
        raise ValueError("memberships must be a 2D array")
    if not baseline_sample_indices:
        raise ValueError("baseline_sample_indices must not be empty")

    params = hyperparams or PriorGraphHyperparams()
    intent = user_intent or PriorGraphUserIntent()
    baseline_mixture = estimate_baseline_mixture(membership_matrix, baseline_sample_indices)
    pool_mixture = membership_matrix.mean(axis=0, dtype=np.float32)

    layouts = _build_block_layouts(feature_groups, feature_label_map)
    default_block_weights = _dimension_balanced_defaults(feature_groups.keys())

    pool_weights = np.ones((membership_matrix.shape[0],), dtype=np.float32)
    baseline_weights = np.zeros((membership_matrix.shape[0],), dtype=np.float32)
    baseline_weights[np.asarray(baseline_sample_indices, dtype=np.int64)] = 1.0

    block_context_rows: list[_BlockContexts] = []
    branch_norms: list[float] = []
    for block_name, matrix in feature_groups.items():
        layout = layouts[block_name]
        shape_matrix, scalar_matrix = _split_matrix(matrix, layout)
        slice_shape = _weighted_slice_average(shape_matrix, membership_matrix, params.epsilon)
        slice_scalar = _weighted_slice_average(scalar_matrix, membership_matrix, params.epsilon)

        pool_shape = _weighted_average(shape_matrix, pool_weights, params.epsilon)
        base_shape = _weighted_average(shape_matrix, baseline_weights, params.epsilon)
        pool_scalar = _weighted_average(scalar_matrix, pool_weights, params.epsilon)
        base_scalar = _weighted_average(scalar_matrix, baseline_weights, params.epsilon)

        cumulative_shape_matrix = _cumulative_rows(shape_matrix)
        cumulative_pool_shape = _cumulative_rows(pool_shape).reshape(-1)
        cumulative_base_shape = _cumulative_rows(base_shape).reshape(-1)
        cumulative_slice_shape = _cumulative_rows(slice_shape).astype(np.float32)

        shape_scale = _robust_scale(cumulative_shape_matrix, params.epsilon)
        scalar_scale = _robust_scale(scalar_matrix, params.epsilon)
        delta_shape = _standardize(
            cumulative_pool_shape - cumulative_base_shape,
            shape_scale,
            params.epsilon,
        )
        delta_scalar = _standardize(
            pool_scalar - base_scalar,
            scalar_scale,
            params.epsilon,
        )

        shape_weights = _soft_saliency(delta_shape, FIXED_SALIENCY_GAMMA, params.epsilon)
        scalar_weights = _soft_saliency(delta_scalar, FIXED_SALIENCY_GAMMA, params.epsilon)
        branch_norms.append(_weighted_norm(delta_shape, shape_weights, params.epsilon))
        branch_norms.append(_weighted_norm(delta_scalar, scalar_weights, params.epsilon))

        rho = _resolve_block_rho(
            has_shape=bool(layout.shape_indices.size),
            has_scalar=bool(layout.scalar_indices.size),
            shape_rho=float(params.shape_rho),
        )

        block_context_rows.append(
            _BlockContexts(
                layout=layout,
                shape_delta=_BranchDelta(delta=delta_shape, weights=shape_weights, scale=shape_scale),
                scalar_delta=_BranchDelta(delta=delta_scalar, weights=scalar_weights, scale=scalar_scale),
                slice_shape=cumulative_slice_shape,
                slice_scalar=slice_scalar,
                pool_shape=cumulative_pool_shape.astype(np.float32),
                pool_scalar=pool_scalar.astype(np.float32),
                rho=rho,
                weight=_resolve_user_block_weight(
                    block_name=block_name,
                    layout=layout,
                    default_weight=float(default_block_weights.get(block_name, 1.0)),
                    user_intent=intent,
                ),
            )
        )

    nonzero_norms = [norm for norm in branch_norms if norm > params.epsilon]
    tau_gap = float(np.quantile(nonzero_norms, 0.25)) if nonzero_norms else 0.25
    block_contexts = tuple(block_context_rows)
    baseline_gap = _compute_portrait_residual_gap_from_contexts(
        mixture=baseline_mixture,
        block_contexts=block_contexts,
        epsilon=params.epsilon,
    )
    return PortraitResidualContext(
        baseline_mixture=np.asarray(baseline_mixture, dtype=np.float32),
        pool_mixture=np.asarray(pool_mixture, dtype=np.float32),
        block_contexts=block_contexts,
        tau_gap=tau_gap,
        baseline_gap=baseline_gap,
        epsilon=float(params.epsilon),
    )


def build_pool_target_portrait_spec(
    *,
    feature_groups: dict[str, np.ndarray],
    feature_label_map: dict[str, list[str]],
    memberships: np.ndarray,
    hyperparams: PriorGraphHyperparams | None = None,
) -> TargetPortraitSpec:
    params = hyperparams or PriorGraphHyperparams()
    membership_matrix = np.asarray(memberships, dtype=np.float32)
    if membership_matrix.ndim != 2:
        raise ValueError("memberships must be a 2D array")

    layouts = _build_block_layouts(feature_groups, feature_label_map)
    default_block_weights = _dimension_balanced_defaults(layouts.keys())
    pool_weights = np.ones((membership_matrix.shape[0],), dtype=np.float32)
    shape_targets: dict[str, np.ndarray] = {}
    scalar_targets: dict[str, np.ndarray] = {}
    included_blocks: set[str] = set()
    for block_name, matrix in feature_groups.items():
        layout = layouts[block_name]
        shape_matrix, scalar_matrix = _split_matrix(matrix, layout)
        if layout.shape_indices.size:
            pooled_shape = _weighted_average(shape_matrix, pool_weights, params.epsilon)
            shape_targets[block_name] = _normalize_shape_target(pooled_shape, layout, params.epsilon)
            included_blocks.add(block_name)
        if layout.scalar_indices.size:
            scalar_targets[block_name] = _weighted_average(scalar_matrix, pool_weights, params.epsilon).astype(np.float32)
            included_blocks.add(block_name)

    return TargetPortraitSpec(
        shape_targets=shape_targets,
        scalar_targets=scalar_targets,
        block_weights={name: float(default_block_weights.get(name, 1.0)) for name in included_blocks},
        source="pool_initialized",
    )


def _extract_support_empty_risks(
    *,
    layout: _BlockLayout,
    slice_scalar: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    if slice_scalar.size == 0:
        return (
            np.zeros((slice_scalar.shape[0],), dtype=np.float32),
            np.zeros((slice_scalar.shape[0],), dtype=np.float32),
        )

    positions = _scalar_label_positions(layout)
    support_risk = np.zeros((slice_scalar.shape[0],), dtype=np.float32)
    empty_flag = np.zeros((slice_scalar.shape[0],), dtype=np.float32)

    if "log_num_values" in positions:
        support_values = np.asarray(slice_scalar[:, positions["log_num_values"]], dtype=np.float32)
        max_support = float(np.max(support_values)) if support_values.size else 0.0
        if max_support > epsilon:
            support_risk = (1.0 - np.clip(support_values / max_support, 0.0, 1.0)).astype(np.float32)
    if "empty_flag" in positions:
        empty_flag = np.clip(
            np.asarray(slice_scalar[:, positions["empty_flag"]], dtype=np.float32),
            0.0,
            1.0,
        ).astype(np.float32)
    return support_risk, empty_flag


def _resolve_target_block_weight(
    *,
    block_name: str,
    default_weight: float,
    target_spec: TargetPortraitSpec,
) -> float:
    return float(target_spec.block_weights.get(block_name, default_weight))


def build_target_residual_context(
    *,
    feature_groups: dict[str, np.ndarray],
    feature_label_map: dict[str, list[str]],
    memberships: np.ndarray,
    baseline_sample_indices: list[int],
    target_spec: TargetPortraitSpec,
    hyperparams: PriorGraphHyperparams | None = None,
) -> TargetResidualContext:
    membership_matrix = np.asarray(memberships, dtype=np.float32)
    if membership_matrix.ndim != 2:
        raise ValueError("memberships must be a 2D array")
    if not baseline_sample_indices:
        raise ValueError("baseline_sample_indices must not be empty")

    params = hyperparams or PriorGraphHyperparams()
    layouts = _build_block_layouts(feature_groups, feature_label_map)
    baseline_weights = np.zeros((membership_matrix.shape[0],), dtype=np.float32)
    baseline_weights[np.asarray(baseline_sample_indices, dtype=np.int64)] = 1.0
    baseline_mixture = estimate_baseline_mixture(membership_matrix, baseline_sample_indices)
    target_blocks = sorted(set(target_spec.shape_targets.keys()) | set(target_spec.scalar_targets.keys()))
    default_block_weights = _dimension_balanced_defaults(target_blocks)

    block_contexts: list[_TargetBlockContext] = []
    branch_norms: list[float] = []
    target_by_block: dict[str, np.ndarray] = {}
    scalar_target_by_block: dict[str, np.ndarray] = {}
    block_weights: dict[str, float] = {}
    slice_shape_by_block: dict[str, np.ndarray] = {}
    slice_scalar_by_block: dict[str, np.ndarray] = {}
    for block_name in target_blocks:
        if block_name not in feature_groups:
            continue
        layout = layouts[block_name]
        shape_matrix, scalar_matrix = _split_matrix(feature_groups[block_name], layout)
        has_shape_target = bool(layout.shape_indices.size) and block_name in target_spec.shape_targets
        has_scalar_target = bool(layout.scalar_indices.size) and block_name in target_spec.scalar_targets
        if not has_shape_target and not has_scalar_target:
            continue

        slice_scalar = _weighted_slice_average(scalar_matrix, membership_matrix, params.epsilon)
        support_risk, empty_flag = _extract_support_empty_risks(
            layout=layout,
            slice_scalar=slice_scalar,
            epsilon=params.epsilon,
        )
        weight = _resolve_target_block_weight(
            block_name=block_name,
            default_weight=float(default_block_weights.get(block_name, 1.0)),
            target_spec=target_spec,
        )

        if has_shape_target:
            slice_shape = _weighted_slice_average(shape_matrix, membership_matrix, params.epsilon)
            baseline_shape = _weighted_average(shape_matrix, baseline_weights, params.epsilon)
            target_shape = _normalize_shape_target(target_spec.shape_targets[block_name], layout, params.epsilon)
            cumulative_shape_matrix = _cumulative_rows(shape_matrix)
            cumulative_slice_shape = _cumulative_rows(slice_shape).astype(np.float32)
            cumulative_baseline_shape = _cumulative_rows(baseline_shape).reshape(-1).astype(np.float32)
            cumulative_target_shape = _cumulative_rows(target_shape).reshape(-1).astype(np.float32)
            shape_scale = _robust_scale(cumulative_shape_matrix, params.epsilon)
            delta_shape = _standardize(cumulative_target_shape - cumulative_baseline_shape, shape_scale, params.epsilon)
            shape_weights = _soft_saliency(delta_shape, FIXED_SALIENCY_GAMMA, params.epsilon)
            target_by_block[block_name] = cumulative_target_shape.astype(np.float32)
            slice_shape_by_block[block_name] = cumulative_slice_shape.astype(np.float32)
            branch_norms.append(_weighted_norm(delta_shape, shape_weights, params.epsilon))
        else:
            cumulative_slice_shape = np.zeros((membership_matrix.shape[1], 0), dtype=np.float32)
            cumulative_baseline_shape = np.zeros((0,), dtype=np.float32)
            cumulative_target_shape = np.zeros((0,), dtype=np.float32)
            shape_scale = np.zeros((0,), dtype=np.float32)
            delta_shape = np.zeros((0,), dtype=np.float32)
            shape_weights = np.zeros((0,), dtype=np.float32)

        if has_scalar_target:
            baseline_scalar = _weighted_average(scalar_matrix, baseline_weights, params.epsilon)
            target_scalar = np.asarray(target_spec.scalar_targets[block_name], dtype=np.float32).reshape(-1)
            scalar_scale = _robust_scale(scalar_matrix, params.epsilon)
            delta_scalar = _standardize(target_scalar - baseline_scalar, scalar_scale, params.epsilon)
            scalar_weights = _soft_saliency(delta_scalar, FIXED_SALIENCY_GAMMA, params.epsilon)
            scalar_target_by_block[block_name] = target_scalar.astype(np.float32)
            slice_scalar_by_block[block_name] = slice_scalar.astype(np.float32)
            branch_norms.append(_weighted_norm(delta_scalar, scalar_weights, params.epsilon))
        else:
            baseline_scalar = np.zeros((0,), dtype=np.float32)
            target_scalar = np.zeros((0,), dtype=np.float32)
            scalar_scale = np.zeros((0,), dtype=np.float32)
            delta_scalar = np.zeros((0,), dtype=np.float32)
            scalar_weights = np.zeros((0,), dtype=np.float32)

        block_weights[block_name] = weight
        block_contexts.append(
            _TargetBlockContext(
                layout=layout,
                weight=weight,
                target_shape=cumulative_target_shape.astype(np.float32),
                target_scalar=target_scalar.astype(np.float32),
                baseline_shape=cumulative_baseline_shape.astype(np.float32),
                baseline_scalar=baseline_scalar.astype(np.float32),
                slice_shape=cumulative_slice_shape.astype(np.float32),
                slice_scalar=slice_scalar.astype(np.float32) if has_scalar_target else np.zeros((membership_matrix.shape[1], 0), dtype=np.float32),
                shape_delta=_BranchDelta(
                    delta=delta_shape.astype(np.float32),
                    weights=shape_weights.astype(np.float32),
                    scale=shape_scale.astype(np.float32),
                ),
                scalar_delta=_BranchDelta(
                    delta=delta_scalar.astype(np.float32),
                    weights=scalar_weights.astype(np.float32),
                    scale=scalar_scale.astype(np.float32),
                ),
                rho=_resolve_block_rho(
                    has_shape=has_shape_target,
                    has_scalar=has_scalar_target,
                    shape_rho=float(params.shape_rho),
                ),
                support_risk_by_slice=support_risk.astype(np.float32),
                empty_flag_by_slice=empty_flag.astype(np.float32),
            )
        )

    nonzero_norms = [norm for norm in branch_norms if norm > params.epsilon]
    tau_gap = float(np.quantile(nonzero_norms, 0.25)) if nonzero_norms else 0.25
    context = TargetResidualContext(
        baseline_mixture=np.asarray(baseline_mixture, dtype=np.float32),
        baseline_gap=0.0,
        target_by_block=target_by_block,
        block_weights=block_weights,
        slice_shape_by_block=slice_shape_by_block,
        scalar_target_by_block=scalar_target_by_block,
        slice_scalar_by_block=slice_scalar_by_block,
        block_contexts=tuple(block_contexts),
        tau_gap=tau_gap,
        epsilon=float(params.epsilon),
    )
    baseline_gap = compute_target_residual_gap(context=context, mixture=baseline_mixture)
    return TargetResidualContext(
        baseline_mixture=np.asarray(baseline_mixture, dtype=np.float32),
        baseline_gap=float(baseline_gap),
        target_by_block=target_by_block,
        block_weights=block_weights,
        slice_shape_by_block=slice_shape_by_block,
        scalar_target_by_block=scalar_target_by_block,
        slice_scalar_by_block=slice_scalar_by_block,
        block_contexts=tuple(block_contexts),
        tau_gap=tau_gap,
        epsilon=float(params.epsilon),
    )


def compute_target_residual_gap(
    *,
    context: TargetResidualContext,
    mixture: np.ndarray,
) -> float:
    weights = np.asarray(mixture, dtype=np.float32).reshape(-1)
    epsilon = float(context.epsilon)
    if context.block_contexts:
        weighted_gap_sum = 0.0
        weight_sum = 0.0
        for block_context in context.block_contexts:
            expected_shape = weights @ block_context.slice_shape if block_context.slice_shape.size else np.zeros((0,), dtype=np.float32)
            expected_scalar = weights @ block_context.slice_scalar if block_context.slice_scalar.size else np.zeros((0,), dtype=np.float32)
            shape_residual = _standardize(block_context.target_shape - expected_shape, block_context.shape_delta.scale, epsilon)
            scalar_residual = _standardize(block_context.target_scalar - expected_scalar, block_context.scalar_delta.scale, epsilon)
            shape_gap = _weighted_norm(shape_residual, block_context.shape_delta.weights, epsilon)
            scalar_gap = _weighted_norm(scalar_residual, block_context.scalar_delta.weights, epsilon)
            block_gap = float(block_context.rho * shape_gap + (1.0 - block_context.rho) * scalar_gap)
            weighted_gap_sum += block_context.weight * block_gap
            weight_sum += block_context.weight
        return float(weighted_gap_sum / (weight_sum + epsilon))

    weighted_gap_sum = 0.0
    weight_sum = 0.0
    block_names = sorted(set(context.target_by_block.keys()) | set(context.scalar_target_by_block.keys()))
    for block_name in block_names:
        target_shape = np.asarray(context.target_by_block.get(block_name, np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        target_scalar = np.asarray(context.scalar_target_by_block.get(block_name, np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        slice_shape = np.asarray(context.slice_shape_by_block.get(block_name, np.zeros((weights.shape[0], 0), dtype=np.float32)), dtype=np.float32)
        slice_scalar = np.asarray(context.slice_scalar_by_block.get(block_name, np.zeros((weights.shape[0], 0), dtype=np.float32)), dtype=np.float32)
        has_shape = bool(target_shape.size and slice_shape.size)
        has_scalar = bool(target_scalar.size and slice_scalar.size)
        if not has_shape and not has_scalar:
            continue
        shape_gap = float(np.linalg.norm(target_shape - (weights @ slice_shape))) if has_shape else 0.0
        scalar_gap = float(np.linalg.norm(target_scalar - (weights @ slice_scalar))) if has_scalar else 0.0
        block_gap = float(_resolve_block_rho(has_shape=has_shape, has_scalar=has_scalar, shape_rho=0.8) * shape_gap)
        if has_shape and has_scalar:
            block_gap += float((1.0 - 0.8) * scalar_gap)
        elif has_scalar and not has_shape:
            block_gap = scalar_gap
        block_weight = float(context.block_weights.get(block_name, 1.0))
        weighted_gap_sum += block_weight * block_gap
        weight_sum += block_weight
    return float(weighted_gap_sum / (weight_sum + epsilon))


def _soft_bias_bonus(donor: str, receiver: str, bias: SearchBias) -> float:
    score = 0.0
    if (donor, receiver) in bias.preferred_edges:
        score += 1.0
    if (donor, receiver) in bias.avoided_edges:
        score -= 1.0
    if receiver in bias.increase_slices:
        score += 0.5
    if donor in bias.increase_slices:
        score -= 0.5
    if donor in bias.decrease_slices:
        score += 0.5
    if receiver in bias.decrease_slices:
        score -= 0.5
    if donor in bias.stable_slices:
        score -= 0.5
    if receiver in bias.stable_slices:
        score -= 0.5
    return float(score)


def _target_mask_reason(
    *,
    donor: str,
    receiver: str,
    donor_index: int,
    receiver_index: int,
    baseline_mixture: np.ndarray,
    constraints: SearchConstraints,
    hyperparams: PriorGraphHyperparams,
) -> tuple[bool, str | None, float, float]:
    if donor == receiver:
        return False, "self_edge", 0.0, 0.0
    if donor in constraints.frozen_slices or receiver in constraints.frozen_slices:
        return False, "frozen_slice", 0.0, 0.0
    if (donor, receiver) in constraints.forbidden_edges:
        return False, "forbidden_edge", 0.0, 0.0

    donor_mass = float(baseline_mixture[donor_index])
    donor_floor = max(hyperparams.min_transfer_mass, hyperparams.donor_keep_ratio * donor_mass)
    donor_transferable = max(0.0, donor_mass - donor_floor)
    receiver_cap = min(1.0, float(baseline_mixture[receiver_index]) + hyperparams.receiver_headroom)
    receiver_slack = max(0.0, receiver_cap - float(baseline_mixture[receiver_index]))
    if donor_transferable <= hyperparams.epsilon:
        return False, "donor_too_small", donor_transferable, receiver_slack
    if receiver_slack <= hyperparams.epsilon:
        return False, "receiver_at_cap", donor_transferable, receiver_slack
    return True, None, donor_transferable, receiver_slack


def _target_side_protected(constraints: SearchConstraints, context: _TargetBlockContext) -> bool:
    return (
        context.layout.name in constraints.protected_blocks
        or context.layout.dimension in constraints.protected_dimensions
    )


def build_target_prior_graph(
    *,
    feature_groups: dict[str, np.ndarray],
    feature_label_map: dict[str, list[str]],
    memberships: np.ndarray,
    baseline_sample_indices: list[int],
    target_spec: TargetPortraitSpec,
    constraints: SearchConstraints | None = None,
    bias: SearchBias | None = None,
    hyperparams: PriorGraphHyperparams | None = None,
    slice_ids: list[str] | None = None,
    target_context: TargetResidualContext | None = None,
    baseline_seed: int | None = None,
    budget: int | None = None,
) -> PriorGraphPayload:
    membership_matrix = np.asarray(memberships, dtype=np.float32)
    if membership_matrix.ndim != 2:
        raise ValueError("memberships must be a 2D array")
    if not baseline_sample_indices:
        raise ValueError("baseline_sample_indices must not be empty")

    params = hyperparams or PriorGraphHyperparams()
    resolved_constraints = constraints or SearchConstraints()
    resolved_bias = bias or SearchBias()
    num_slices = int(membership_matrix.shape[1])
    resolved_slice_ids = slice_ids or [f"slice_{index:02d}" for index in range(num_slices)]
    if len(resolved_slice_ids) != num_slices:
        raise ValueError("slice_ids length must match membership columns")

    resolved_target_context = target_context or build_target_residual_context(
        feature_groups=feature_groups,
        feature_label_map=feature_label_map,
        memberships=membership_matrix,
        baseline_sample_indices=baseline_sample_indices,
        target_spec=target_spec,
        hyperparams=params,
    )
    baseline_mixture = resolved_target_context.baseline_mixture

    nodes = [
        PriorGraphNode(
            slice_id=slice_id,
            index=index,
            canonical_weight=float(baseline_mixture[index]),
            baseline_weight=float(baseline_mixture[index]),
            pool_delta=0.0,
            instability_score=0.0,
            default_action_state="inspect",
        )
        for index, slice_id in enumerate(resolved_slice_ids)
    ]

    edges: list[PriorGraphEdge] = []
    block_contexts = tuple(resolved_target_context.block_contexts)
    for donor_index, donor in enumerate(resolved_slice_ids):
        for receiver_index, receiver in enumerate(resolved_slice_ids):
            if donor == receiver:
                continue

            admissible, masked_reason, donor_transferable, receiver_slack = _target_mask_reason(
                donor=donor,
                receiver=receiver,
                donor_index=donor_index,
                receiver_index=receiver_index,
                baseline_mixture=baseline_mixture,
                constraints=resolved_constraints,
                hyperparams=params,
            )

            block_scores: dict[str, float] = {}
            weighted_block_sum = 0.0
            weight_sum = 0.0
            side_risk = 0.0
            support_empty_weighted = 0.0
            support_empty_weight_sum = 0.0
            for context in block_contexts:
                shape_effect = _standardize(
                    context.slice_shape[receiver_index] - context.slice_shape[donor_index],
                    context.shape_delta.scale,
                    params.epsilon,
                )
                scalar_effect = _standardize(
                    context.slice_scalar[receiver_index] - context.slice_scalar[donor_index],
                    context.scalar_delta.scale,
                    params.epsilon,
                )
                shape_score = _branch_score(
                    context.shape_delta.delta,
                    shape_effect,
                    context.shape_delta.weights,
                    tau_gap=resolved_target_context.tau_gap,
                    epsilon=params.epsilon,
                )
                scalar_score = _branch_score(
                    context.scalar_delta.delta,
                    scalar_effect,
                    context.scalar_delta.weights,
                    tau_gap=resolved_target_context.tau_gap,
                    epsilon=params.epsilon,
                )
                block_score = float(context.rho * shape_score + (1.0 - context.rho) * scalar_score)
                block_scores[context.layout.name] = float(block_score)
                weighted_block_sum += context.weight * block_score
                weight_sum += context.weight
                if _target_side_protected(resolved_constraints, context):
                    side_risk += max(0.0, -float(block_score))
                support_empty = 0.25 * (
                    float(context.support_risk_by_slice[donor_index])
                    + float(context.support_risk_by_slice[receiver_index])
                    + float(context.empty_flag_by_slice[donor_index])
                    + float(context.empty_flag_by_slice[receiver_index])
                )
                support_empty_weighted += context.weight * support_empty
                support_empty_weight_sum += context.weight

            fit_score = float(weighted_block_sum / (weight_sum + params.epsilon))
            bias_score = _soft_bias_bonus(donor, receiver, resolved_bias)
            donor_margin = float(
                donor_transferable / (max(float(baseline_mixture[donor_index]), params.min_transfer_mass) + params.epsilon)
            )
            receiver_margin = float(
                receiver_slack / (min(1.0, float(baseline_mixture[receiver_index]) + params.receiver_headroom) + params.epsilon)
            )
            boundary_risk = float(1.0 - np.clip(min(donor_margin, receiver_margin), 0.0, 1.0))
            support_empty_risk = float(support_empty_weighted / (support_empty_weight_sum + params.epsilon))
            risk_components = {
                "side_risk": float(side_risk),
                "boundary_risk": float(boundary_risk),
                "support_empty_risk": float(support_empty_risk),
            }
            risk_score = float(sum(risk_components.values()) / 3.0)
            total_score = float(fit_score - params.lambda_risk * risk_score + bias_score)
            amplitude_band = (
                0.0 if not admissible else float(min(params.min_transfer_mass, donor_transferable, receiver_slack)),
                0.0 if not admissible else float(min(donor_transferable, receiver_slack)),
            )
            edges.append(
                PriorGraphEdge(
                    edge_id=f"{donor}__to__{receiver}",
                    donor=donor,
                    receiver=receiver,
                    admissible=admissible,
                    masked_reason=masked_reason,
                    score=total_score,
                    fit_score=fit_score,
                    balance_score=0.0,
                    user_score=0.0,
                    risk_score=risk_score,
                    block_scores=block_scores,
                    amplitude_band=amplitude_band,
                    visible_by_default=False,
                    bias_score=bias_score,
                    risk_components=risk_components,
                )
            )

    ranked_visible = [
        edge.edge_id
        for edge in sorted(
            (edge for edge in edges if edge.admissible and edge.score >= params.score_threshold),
            key=lambda item: item.score,
            reverse=True,
        )[: params.top_k_render]
    ]
    visible_ids = set(ranked_visible)
    finalized_edges = [
        PriorGraphEdge(
            edge_id=edge.edge_id,
            donor=edge.donor,
            receiver=edge.receiver,
            admissible=edge.admissible,
            masked_reason=edge.masked_reason,
            score=edge.score,
            fit_score=edge.fit_score,
            balance_score=edge.balance_score,
            user_score=edge.user_score,
            risk_score=edge.risk_score,
            block_scores=edge.block_scores,
            amplitude_band=edge.amplitude_band,
            visible_by_default=edge.edge_id in visible_ids,
            bias_score=edge.bias_score,
            risk_components=dict(edge.risk_components),
        )
        for edge in edges
    ]

    return PriorGraphPayload(
        nodes=nodes,
        edges=finalized_edges,
        graph_context={
            "baseline_seed": baseline_seed,
            "budget": budget,
            "target_blocks": sorted(set(target_spec.shape_targets.keys()) | set(target_spec.scalar_targets.keys())),
            "target_source": target_spec.source,
        },
        defaults={
            "top_k_render": params.top_k_render,
            "score_threshold": params.score_threshold,
            "shape_rho": params.shape_rho,
            "lambda_risk": params.lambda_risk,
        },
    )


def build_prior_graph(
    *,
    feature_groups: dict[str, np.ndarray],
    feature_label_map: dict[str, list[str]],
    memberships: np.ndarray,
    baseline_sample_indices: list[int],
    slice_ids: list[str] | None = None,
    hyperparams: PriorGraphHyperparams | None = None,
    user_intent: PriorGraphUserIntent | None = None,
    portrait_context: PortraitResidualContext | None = None,
    baseline_seed: int | None = None,
    budget: int | None = None,
) -> PriorGraphPayload:
    membership_matrix = np.asarray(memberships, dtype=np.float32)
    if membership_matrix.ndim != 2:
        raise ValueError("memberships must be a 2D array")
    if not baseline_sample_indices:
        raise ValueError("baseline_sample_indices must not be empty")

    params = hyperparams or PriorGraphHyperparams()
    intent = user_intent or PriorGraphUserIntent()
    num_slices = int(membership_matrix.shape[1])
    resolved_slice_ids = slice_ids or [f"slice_{index}" for index in range(num_slices)]
    if len(resolved_slice_ids) != num_slices:
        raise ValueError("slice_ids length must match membership columns")

    resolved_portrait_context = portrait_context or build_portrait_residual_context(
        feature_groups=feature_groups,
        feature_label_map=feature_label_map,
        memberships=membership_matrix,
        baseline_sample_indices=baseline_sample_indices,
        hyperparams=params,
        user_intent=intent,
    )
    baseline_mixture = resolved_portrait_context.baseline_mixture
    pool_mixture = resolved_portrait_context.pool_mixture
    slice_instability = _compute_slice_instability(membership_matrix, params.epsilon)
    block_contexts = {context.layout.name: context for context in resolved_portrait_context.block_contexts}
    tau_gap = float(resolved_portrait_context.tau_gap)

    nodes = [
        PriorGraphNode(
            slice_id=slice_id,
            index=index,
            canonical_weight=float(pool_mixture[index]),
            baseline_weight=float(baseline_mixture[index]),
            pool_delta=float(pool_mixture[index] - baseline_mixture[index]),
            instability_score=float(slice_instability[index]),
            default_action_state=_default_action_state(float(pool_mixture[index] - baseline_mixture[index]), params.min_transfer_mass),
        )
        for index, slice_id in enumerate(resolved_slice_ids)
    ]

    edges: list[PriorGraphEdge] = []
    for donor_index, donor in enumerate(resolved_slice_ids):
        for receiver_index, receiver in enumerate(resolved_slice_ids):
            if donor == receiver:
                continue

            admissible, masked_reason, donor_transferable, receiver_slack = _mask_reason(
                donor=donor,
                receiver=receiver,
                donor_index=donor_index,
                receiver_index=receiver_index,
                baseline_mixture=baseline_mixture,
                pool_mixture=pool_mixture,
                user_intent=intent,
                hyperparams=params,
            )

            block_scores: dict[str, float] = {}
            weighted_block_sum = 0.0
            weight_sum = 0.0
            side_risk = 0.0
            for block_name, context in block_contexts.items():
                shape_effect = _standardize(
                    context.slice_shape[receiver_index] - context.slice_shape[donor_index],
                    context.shape_delta.scale,
                    params.epsilon,
                )
                scalar_effect = _standardize(
                    context.slice_scalar[receiver_index] - context.slice_scalar[donor_index],
                    context.scalar_delta.scale,
                    params.epsilon,
                )
                shape_score = _branch_score(
                    context.shape_delta.delta,
                    shape_effect,
                    context.shape_delta.weights,
                    tau_gap,
                    params.epsilon,
                )
                scalar_score = _branch_score(
                    context.scalar_delta.delta,
                    scalar_effect,
                    context.scalar_delta.weights,
                    tau_gap,
                    params.epsilon,
                )
                block_score = float(context.rho * shape_score + (1.0 - context.rho) * scalar_score)
                block_scores[block_name] = block_score
                weighted_block_sum += context.weight * block_score
                weight_sum += context.weight
                if _protected_blocks(intent, block_name, context.layout):
                    side_risk += max(0.0, -block_score)

            fit_score = float(weighted_block_sum / (weight_sum + params.epsilon))
            receiver_need = max(0.0, float(pool_mixture[receiver_index] - baseline_mixture[receiver_index]))
            donor_surplus = max(0.0, float(baseline_mixture[donor_index] - pool_mixture[donor_index]))
            balance_score = float(receiver_need + donor_surplus)
            user_score = _user_score(donor, receiver, intent)
            instability_risk = float((slice_instability[donor_index] + slice_instability[receiver_index]) / 2.0)
            donor_margin = float(donor_transferable / (max(float(baseline_mixture[donor_index]), params.min_transfer_mass) + params.epsilon))
            receiver_margin = float(receiver_slack / (min(1.0, float(pool_mixture[receiver_index]) + params.receiver_headroom) + params.epsilon))
            extreme_risk = float(1.0 - np.clip(min(donor_margin, receiver_margin), 0.0, 1.0))
            risk_score = float((side_risk + instability_risk + extreme_risk) / 3.0)
            total_score = float(
                fit_score
                + params.lambda_user * user_score
                - params.lambda_risk * risk_score
            )

            amplitude_band = (
                0.0 if not admissible else float(min(params.min_transfer_mass, donor_transferable, receiver_slack)),
                0.0 if not admissible else float(min(donor_transferable, receiver_slack)),
            )
            edges.append(
                PriorGraphEdge(
                    edge_id=f"{donor}__to__{receiver}",
                    donor=donor,
                    receiver=receiver,
                    admissible=admissible,
                    masked_reason=masked_reason,
                    score=total_score,
                    fit_score=fit_score,
                    balance_score=balance_score,
                    user_score=user_score,
                    risk_score=risk_score,
                    block_scores=block_scores,
                    amplitude_band=amplitude_band,
                    visible_by_default=False,
                )
            )

    ranked_visible = [
        edge.edge_id
        for edge in sorted(
            (
                edge
                for edge in edges
                if edge.admissible and edge.score >= params.score_threshold
            ),
            key=lambda item: item.score,
            reverse=True,
        )[: params.top_k_render]
    ]
    visible_ids = set(ranked_visible)
    finalized_edges = [
        PriorGraphEdge(
            edge_id=edge.edge_id,
            donor=edge.donor,
            receiver=edge.receiver,
            admissible=edge.admissible,
            masked_reason=edge.masked_reason,
            score=edge.score,
            fit_score=edge.fit_score,
            balance_score=edge.balance_score,
            user_score=edge.user_score,
            risk_score=edge.risk_score,
            block_scores=edge.block_scores,
            amplitude_band=edge.amplitude_band,
            visible_by_default=edge.edge_id in visible_ids,
        )
        for edge in edges
    ]

    return PriorGraphPayload(
        nodes=nodes,
        edges=finalized_edges,
        graph_context={
            "baseline_seed": baseline_seed,
            "budget": budget,
            "dimension_weights": {dimension: float(intent.dimension_weights.get(dimension, 1.0)) for dimension in DIMENSIONS},
            "atomic_block_weights": {block_name: float(intent.atomic_block_weights.get(block_name, 1.0)) for block_name in feature_groups},
            "tau_gap": tau_gap,
        },
        defaults={
            "top_k_render": params.top_k_render,
            "score_threshold": params.score_threshold,
            "shape_rho": params.shape_rho,
            "lambda_balance": params.lambda_balance,
            "lambda_user": params.lambda_user,
            "lambda_risk": params.lambda_risk,
        },
    )
