from __future__ import annotations

from typing import Iterable

import numpy as np


def build_sample_index(sample_ids: list[str]) -> dict[str, int]:
    return {str(sample_id): int(index) for index, sample_id in enumerate(sample_ids)}


def resolve_sample_indices(sample_index: dict[str, int], selected_ids: Iterable[str]) -> list[int]:
    indices: list[int] = []
    for sample_id in selected_ids:
        key = str(sample_id)
        if key not in sample_index:
            raise KeyError(f"sample_id '{key}' not found in sample index")
        indices.append(int(sample_index[key]))
    return indices


def aggregate_feature_groups_by_indices(
    feature_groups: dict[str, np.ndarray],
    sample_indices: list[int],
) -> dict[str, np.ndarray]:
    if not sample_indices:
        raise ValueError("sample_indices must not be empty")

    selected = np.asarray(sample_indices, dtype=np.int64)
    aggregated: dict[str, np.ndarray] = {}
    for block_name, matrix in feature_groups.items():
        group_matrix = np.asarray(matrix, dtype=np.float32)
        if group_matrix.ndim == 1:
            group_matrix = group_matrix[:, None]
        aggregated[block_name] = group_matrix[selected].mean(axis=0, dtype=np.float32).reshape(-1).astype(np.float32)
    return aggregated


def subtract_feature_groups(
    target_features: dict[str, np.ndarray],
    baseline_features: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    names = sorted(set(target_features.keys()) | set(baseline_features.keys()))
    deltas: dict[str, np.ndarray] = {}
    for name in names:
        target = np.asarray(target_features.get(name, []), dtype=np.float32).reshape(-1)
        baseline = np.asarray(baseline_features.get(name, []), dtype=np.float32).reshape(-1)
        if target.shape != baseline.shape:
            raise ValueError(f"feature group '{name}' shape mismatch: {target.shape} vs {baseline.shape}")
        deltas[name] = (target - baseline).astype(np.float32)
    return deltas


def serialize_feature_groups(feature_groups: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {
        block_name: [float(value) for value in np.asarray(vector, dtype=np.float32).reshape(-1).tolist()]
        for block_name, vector in feature_groups.items()
    }


def summarize_feature_groups(feature_groups: dict[str, np.ndarray]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for block_name, vector in feature_groups.items():
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
