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


def sample_budgeted_subset(
    sample_ids: list[str],
    weights: np.ndarray,
    budget: int,
    seed: int,
) -> list[str]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget > len(sample_ids):
        raise ValueError("budget cannot exceed number of available samples")

    probs = np.asarray(weights, dtype=np.float64)
    if probs.ndim != 1 or probs.shape[0] != len(sample_ids):
        raise ValueError("weights must be a 1D array matching sample_ids")
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    probs = probs / total

    rng = np.random.default_rng(int(seed))
    selected = rng.choice(len(sample_ids), size=int(budget), replace=False, p=probs)
    return [sample_ids[int(index)] for index in selected.tolist()]
