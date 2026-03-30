from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MaterializationResult:
    selected_ids: list[str]
    selected_indices: list[int]
    realized_mixture: np.ndarray
    mixture_l1_before_coverage_repair: float | None
    mixture_l1_after_coverage_repair: float
    focus_coverage_before: list[int]
    focus_coverage_after: list[int]
    accepted_coverage_swaps: int


def compute_importance_weights(
    memberships: np.ndarray,
    target_mixture: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    membership_matrix = np.asarray(memberships, dtype=np.float32)
    target = np.asarray(target_mixture, dtype=np.float32)
    if membership_matrix.ndim != 2:
        raise ValueError("memberships must be a 2D array")
    if target.ndim != 1:
        raise ValueError("target_mixture must be a 1D array")
    if membership_matrix.shape[1] != target.shape[0]:
        raise ValueError("target_mixture length must match membership columns")

    pool_mixture = membership_matrix.mean(axis=0, dtype=np.float32)
    reweight = target / np.clip(pool_mixture, eps, None)
    scores = membership_matrix @ reweight
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(scores.sum())
    if total <= 0.0:
        raise ValueError("importance weights must have positive mass")
    return (scores / total).astype(np.float32)


def summarize_target_quotas(target_mixture: np.ndarray, budget: int) -> list[int]:
    target = np.asarray(target_mixture, dtype=np.float32)
    raw = target * float(budget)
    quotas = np.floor(raw).astype(np.int64)
    remainder = int(budget - quotas.sum())
    if remainder > 0:
        order = np.argsort(-(raw - quotas))
        for index in order[:remainder]:
            quotas[index] += 1
    return [int(value) for value in quotas.tolist()]


def _normalize_probabilities(weights: np.ndarray) -> np.ndarray:
    probs = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    return probs / total


def _sample_without_replacement(
    *,
    rng: np.random.Generator,
    candidates: np.ndarray,
    scores: np.ndarray,
    count: int,
) -> np.ndarray:
    if count <= 0 or candidates.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if count >= candidates.size:
        return np.asarray(candidates, dtype=np.int64)
    probs = np.asarray(scores, dtype=np.float64).reshape(-1)
    probs = probs / float(probs.sum())
    local = rng.choice(candidates.size, size=int(count), replace=False, p=probs)
    return np.asarray(candidates[local], dtype=np.int64)


def _objective_distance(current_sum: np.ndarray, target_sum: np.ndarray) -> float:
    return float(np.abs(np.asarray(current_sum, dtype=np.float64) - np.asarray(target_sum, dtype=np.float64)).sum())


def _focus_class_coverage_counts(
    selected_indices: list[int],
    class_presence: np.ndarray,
    focus_class_indices: np.ndarray,
) -> np.ndarray:
    if len(selected_indices) == 0 or focus_class_indices.size == 0:
        return np.zeros((focus_class_indices.size,), dtype=np.int64)
    selected = np.asarray(selected_indices, dtype=np.int64)
    return np.asarray(class_presence[selected][:, focus_class_indices].sum(axis=0), dtype=np.int64)


def build_focus_class_targets(
    class_presence: np.ndarray,
    focus_class_indices: list[int] | np.ndarray,
    budget: int,
    *,
    min_target_count: int = 1,
    max_target_count: int | None = None,
) -> np.ndarray:
    matrix = np.asarray(class_presence, dtype=np.uint8)
    indices = np.asarray(focus_class_indices, dtype=np.int64)
    if indices.size == 0:
        return np.zeros((0,), dtype=np.int64)
    pool_rates = matrix[:, indices].mean(axis=0, dtype=np.float64)
    targets = np.rint(pool_rates * float(budget)).astype(np.int64)
    targets = np.maximum(targets, int(min_target_count))
    if max_target_count is not None:
        targets = np.minimum(targets, int(max_target_count))
    return np.asarray(targets, dtype=np.int64)


def _coverage_gain(
    *,
    current_counts: np.ndarray,
    next_counts: np.ndarray,
    target_counts: np.ndarray,
    class_weights: np.ndarray,
) -> float:
    current_deficit = np.clip(target_counts - current_counts, 0, None)
    next_deficit = np.clip(target_counts - next_counts, 0, None)
    return float(np.dot(class_weights, current_deficit - next_deficit))


def _coverage_repair_subset(
    *,
    selected_indices: list[int],
    selected_mask: np.ndarray,
    membership_matrix: np.ndarray,
    target_sum: np.ndarray,
    class_presence: np.ndarray,
    focus_class_indices: np.ndarray,
    focus_class_targets: np.ndarray,
    focus_class_weights: np.ndarray,
    coverage_alpha: float,
    coverage_repair_budget: int,
) -> tuple[list[int], np.ndarray, int, float]:
    if focus_class_indices.size == 0 or coverage_repair_budget <= 0:
        current_sum = membership_matrix[np.asarray(selected_indices, dtype=np.int64)].sum(axis=0, dtype=np.float64)
        coverage = _focus_class_coverage_counts(selected_indices, class_presence, focus_class_indices)
        return list(selected_indices), coverage, 0, _objective_distance(current_sum, target_sum)

    current_indices = list(selected_indices)
    current_sum = membership_matrix[np.asarray(current_indices, dtype=np.int64)].sum(axis=0, dtype=np.float64)
    current_coverage = _focus_class_coverage_counts(current_indices, class_presence, focus_class_indices)
    current_obj = _objective_distance(current_sum, target_sum)
    accepted = 0

    for _ in range(int(coverage_repair_budget)):
        best_move: tuple[float, int, int, np.ndarray, np.ndarray, float] | None = None
        selected_array = np.asarray(current_indices, dtype=np.int64)
        unselected = np.flatnonzero(~selected_mask)
        if selected_array.size == 0 or unselected.size == 0:
            break

        selected_focus = class_presence[selected_array][:, focus_class_indices]
        unselected_focus = class_presence[unselected][:, focus_class_indices]
        deficit = np.clip(focus_class_targets - current_coverage, 0, None)
        if not np.any(deficit > 0):
            break

        in_scores = (unselected_focus * deficit.reshape(1, -1) * focus_class_weights.reshape(1, -1)).sum(axis=1)
        candidate_in = unselected[np.argsort(-in_scores)[: min(24, len(unselected))]]

        out_scores = (selected_focus * focus_class_weights.reshape(1, -1)).sum(axis=1)
        candidate_out = selected_array[np.argsort(out_scores)[: min(24, len(selected_array))]]

        for in_idx in candidate_in.tolist():
            in_focus = class_presence[int(in_idx), focus_class_indices].astype(np.int64)
            for out_idx in candidate_out.tolist():
                if int(in_idx) == int(out_idx):
                    continue
                out_focus = class_presence[int(out_idx), focus_class_indices].astype(np.int64)
                next_coverage = current_coverage - out_focus + in_focus
                coverage_gain = _coverage_gain(
                    current_counts=current_coverage,
                    next_counts=next_coverage,
                    target_counts=focus_class_targets,
                    class_weights=focus_class_weights,
                )
                if coverage_gain <= 0.0:
                    continue
                next_sum = current_sum - membership_matrix[int(out_idx)].astype(np.float64) + membership_matrix[int(in_idx)].astype(np.float64)
                next_obj = _objective_distance(next_sum, target_sum)
                mixture_damage = max(0.0, next_obj - current_obj)
                score = float(coverage_gain - float(coverage_alpha) * mixture_damage)
                if score <= 0.0:
                    continue
                if best_move is None or score > best_move[0]:
                    best_move = (score, int(in_idx), int(out_idx), next_coverage, next_sum, next_obj)

        if best_move is None:
            break

        _score, in_idx, out_idx, next_coverage, next_sum, next_obj = best_move
        selected_mask[out_idx] = False
        selected_mask[in_idx] = True
        current_indices.remove(out_idx)
        current_indices.append(in_idx)
        current_coverage = np.asarray(next_coverage, dtype=np.int64)
        current_sum = np.asarray(next_sum, dtype=np.float64)
        current_obj = float(next_obj)
        accepted += 1

    return current_indices, current_coverage, accepted, current_obj


def _quota_sample_budgeted_subset(
    sample_ids: list[str],
    weights: np.ndarray,
    memberships: np.ndarray,
    target_mixture: np.ndarray,
    budget: int,
    seed: int,
    *,
    class_presence: np.ndarray | None = None,
    focus_class_indices: list[int] | np.ndarray | None = None,
    focus_class_targets: np.ndarray | None = None,
    focus_class_weights: np.ndarray | None = None,
    coverage_alpha: float = 0.25,
    coverage_repair_budget: int = 64,
) -> MaterializationResult:
    membership_matrix = np.asarray(memberships, dtype=np.float32)
    target = np.asarray(target_mixture, dtype=np.float32).reshape(-1)
    if membership_matrix.ndim != 2:
        raise ValueError("memberships must be a 2D array")
    if membership_matrix.shape[0] != len(sample_ids):
        raise ValueError("memberships row count must match sample_ids")
    if membership_matrix.shape[1] != target.shape[0]:
        raise ValueError("target_mixture length must match membership columns")

    probs = _normalize_probabilities(weights)
    dominant_slice = np.argmax(membership_matrix, axis=1)
    quotas = summarize_target_quotas(target, budget)
    rng = np.random.default_rng(int(seed))

    selected_mask = np.zeros((len(sample_ids),), dtype=bool)
    selected_indices: list[int] = []

    for slice_index, quota in sorted(enumerate(quotas), key=lambda item: item[1], reverse=True):
        if quota <= 0:
            continue
        candidates = np.flatnonzero((dominant_slice == int(slice_index)) & (~selected_mask))
        if candidates.size == 0:
            continue
        scores = probs[candidates] * np.clip(membership_matrix[candidates, int(slice_index)], 1e-12, None)
        take = min(int(quota), int(candidates.size))
        chosen = _sample_without_replacement(rng=rng, candidates=candidates, scores=scores, count=take)
        selected_mask[chosen] = True
        selected_indices.extend(int(index) for index in chosen.tolist())

    remaining = int(budget - len(selected_indices))
    if remaining > 0:
        candidates = np.flatnonzero(~selected_mask)
        chosen = _sample_without_replacement(
            rng=rng,
            candidates=candidates,
            scores=probs[candidates],
            count=remaining,
        )
        selected_mask[chosen] = True
        selected_indices.extend(int(index) for index in chosen.tolist())

    if len(selected_indices) != int(budget):
        raise RuntimeError("materialization did not produce the requested budget")

    target_sum = target.astype(np.float64) * float(budget)
    current_sum = membership_matrix[np.asarray(selected_indices, dtype=np.int64)].sum(axis=0, dtype=np.float64)
    for _ in range(max(8, int(target.shape[0]) * 2)):
        residual = target_sum - current_sum
        receiver = int(np.argmax(residual))
        donor = int(np.argmin(residual))
        if float(residual[receiver]) <= 1e-9 or float(-residual[donor]) <= 1e-9:
            break

        selected_array = np.asarray(selected_indices, dtype=np.int64)
        donor_pool = selected_array[dominant_slice[selected_array] == donor]
        if donor_pool.size == 0:
            donor_pool = selected_array

        unselected_pool = np.flatnonzero(~selected_mask)
        receiver_pool = unselected_pool[dominant_slice[unselected_pool] == receiver]
        if receiver_pool.size == 0:
            receiver_pool = unselected_pool
        if receiver_pool.size == 0:
            break

        donor_scores = membership_matrix[donor_pool, donor] - membership_matrix[donor_pool, receiver]
        receiver_scores = membership_matrix[receiver_pool, receiver] - membership_matrix[receiver_pool, donor]
        donor_idx = int(donor_pool[int(np.argmax(donor_scores))])
        receiver_idx = int(receiver_pool[int(np.argmax(receiver_scores))])

        current_obj = _objective_distance(current_sum, target_sum)
        swapped_sum = current_sum - membership_matrix[donor_idx].astype(np.float64) + membership_matrix[receiver_idx].astype(np.float64)
        swapped_obj = _objective_distance(swapped_sum, target_sum)
        if swapped_obj + 1e-9 >= current_obj:
            break

        selected_mask[donor_idx] = False
        selected_mask[receiver_idx] = True
        selected_indices.remove(donor_idx)
        selected_indices.append(receiver_idx)
        current_sum = swapped_sum

    mixture_l1_before_coverage = _objective_distance(current_sum, target_sum)
    focus_indices = np.asarray(focus_class_indices if focus_class_indices is not None else [], dtype=np.int64)
    focus_before = (
        _focus_class_coverage_counts(selected_indices, np.asarray(class_presence, dtype=np.uint8), focus_indices).tolist()
        if class_presence is not None and focus_indices.size > 0
        else []
    )
    accepted_swaps = 0
    final_obj = mixture_l1_before_coverage
    focus_after = list(focus_before)

    if class_presence is not None and focus_indices.size > 0:
        class_presence_matrix = np.asarray(class_presence, dtype=np.uint8)
        if class_presence_matrix.shape[0] != len(sample_ids):
            raise ValueError("class_presence row count must match sample_ids")
        if focus_class_targets is None:
            focus_class_targets = build_focus_class_targets(class_presence_matrix, focus_indices, budget)
        else:
            focus_class_targets = np.asarray(focus_class_targets, dtype=np.int64)
        if focus_class_weights is None:
            focus_class_weights = np.ones((focus_indices.size,), dtype=np.float32)
        else:
            focus_class_weights = np.asarray(focus_class_weights, dtype=np.float32)

        selected_indices, repaired_coverage, accepted_swaps, final_obj = _coverage_repair_subset(
            selected_indices=selected_indices,
            selected_mask=selected_mask,
            membership_matrix=membership_matrix,
            target_sum=target_sum,
            class_presence=class_presence_matrix,
            focus_class_indices=focus_indices,
            focus_class_targets=focus_class_targets,
            focus_class_weights=focus_class_weights,
            coverage_alpha=float(coverage_alpha),
            coverage_repair_budget=int(coverage_repair_budget),
        )
        focus_after = repaired_coverage.tolist()
        current_sum = membership_matrix[np.asarray(selected_indices, dtype=np.int64)].sum(axis=0, dtype=np.float64)

    return MaterializationResult(
        selected_ids=[sample_ids[int(index)] for index in selected_indices],
        selected_indices=[int(index) for index in selected_indices],
        realized_mixture=(current_sum / float(budget)).astype(np.float32),
        mixture_l1_before_coverage_repair=float(mixture_l1_before_coverage),
        mixture_l1_after_coverage_repair=float(final_obj),
        focus_coverage_before=[int(value) for value in focus_before],
        focus_coverage_after=[int(value) for value in focus_after],
        accepted_coverage_swaps=int(accepted_swaps),
    )


def materialize_budgeted_subset(
    sample_ids: list[str],
    weights: np.ndarray,
    budget: int,
    seed: int,
    *,
    memberships: np.ndarray | None = None,
    target_mixture: np.ndarray | None = None,
    class_presence: np.ndarray | None = None,
    focus_class_indices: list[int] | np.ndarray | None = None,
    focus_class_targets: np.ndarray | None = None,
    focus_class_weights: np.ndarray | None = None,
    coverage_alpha: float = 0.25,
    coverage_repair_budget: int = 64,
) -> MaterializationResult:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget > len(sample_ids):
        raise ValueError("budget cannot exceed number of available samples")

    if memberships is not None and target_mixture is not None:
        return _quota_sample_budgeted_subset(
            sample_ids,
            weights,
            memberships,
            target_mixture,
            budget,
            seed,
            class_presence=class_presence,
            focus_class_indices=focus_class_indices,
            focus_class_targets=focus_class_targets,
            focus_class_weights=focus_class_weights,
            coverage_alpha=coverage_alpha,
            coverage_repair_budget=coverage_repair_budget,
        )

    probs = np.asarray(weights, dtype=np.float64)
    if probs.ndim != 1 or probs.shape[0] != len(sample_ids):
        raise ValueError("weights must be a 1D array matching sample_ids")
    probs = _normalize_probabilities(probs)

    rng = np.random.default_rng(int(seed))
    selected = rng.choice(len(sample_ids), size=int(budget), replace=False, p=probs)
    return MaterializationResult(
        selected_ids=[sample_ids[int(index)] for index in selected.tolist()],
        selected_indices=[int(index) for index in selected.tolist()],
        realized_mixture=np.zeros((0,), dtype=np.float32),
        mixture_l1_before_coverage_repair=None,
        mixture_l1_after_coverage_repair=0.0,
        focus_coverage_before=[],
        focus_coverage_after=[],
        accepted_coverage_swaps=0,
    )


def sample_budgeted_subset(
    sample_ids: list[str],
    weights: np.ndarray,
    budget: int,
    seed: int,
    *,
    memberships: np.ndarray | None = None,
    target_mixture: np.ndarray | None = None,
    class_presence: np.ndarray | None = None,
    focus_class_indices: list[int] | np.ndarray | None = None,
    focus_class_targets: np.ndarray | None = None,
    focus_class_weights: np.ndarray | None = None,
    coverage_alpha: float = 0.25,
    coverage_repair_budget: int = 64,
) -> list[str]:
    return materialize_budgeted_subset(
        sample_ids,
        weights,
        budget,
        seed,
        memberships=memberships,
        target_mixture=target_mixture,
        class_presence=class_presence,
        focus_class_indices=focus_class_indices,
        focus_class_targets=focus_class_targets,
        focus_class_weights=focus_class_weights,
        coverage_alpha=coverage_alpha,
        coverage_repair_budget=coverage_repair_budget,
    ).selected_ids
