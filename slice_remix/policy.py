from __future__ import annotations

import numpy as np


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


def _quota_sample_budgeted_subset(
    sample_ids: list[str],
    weights: np.ndarray,
    memberships: np.ndarray,
    target_mixture: np.ndarray,
    budget: int,
    seed: int,
) -> list[str]:
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

    # First satisfy per-slice quotas using dominant-slice pools to preserve the target mass budget.
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

    # Local repair: swap in/out samples when it decreases mixture distance to the target.
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

    return [sample_ids[int(index)] for index in selected_indices]


def sample_budgeted_subset(
    sample_ids: list[str],
    weights: np.ndarray,
    budget: int,
    seed: int,
    *,
    memberships: np.ndarray | None = None,
    target_mixture: np.ndarray | None = None,
) -> list[str]:
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
        )

    probs = np.asarray(weights, dtype=np.float64)
    if probs.ndim != 1 or probs.shape[0] != len(sample_ids):
        raise ValueError("weights must be a 1D array matching sample_ids")
    probs = _normalize_probabilities(probs)

    rng = np.random.default_rng(int(seed))
    selected = rng.choice(len(sample_ids), size=int(budget), replace=False, p=probs)
    return [sample_ids[int(index)] for index in selected.tolist()]
