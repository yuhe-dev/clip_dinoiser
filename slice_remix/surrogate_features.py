from __future__ import annotations

from typing import Iterable

import numpy as np

from .portraits import build_feature_label_map
from .realized_features import aggregate_feature_groups_by_indices, serialize_feature_groups, summarize_feature_groups


def _safe_probabilities(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    total = float(vector.sum())
    if total <= 0:
        raise ValueError("probability vector must have positive mass")
    return (vector / total).astype(np.float32)


def compute_soft_mixture(memberships: np.ndarray, sample_indices: list[int]) -> np.ndarray:
    matrix = np.asarray(memberships, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("memberships must be a 2D array")
    if not sample_indices:
        raise ValueError("sample_indices must not be empty")
    mixture = matrix[np.asarray(sample_indices, dtype=np.int64)].mean(axis=0, dtype=np.float32)
    return _safe_probabilities(mixture)


def compute_hard_mixture(hard_assignment: np.ndarray, sample_indices: list[int], num_slices: int) -> np.ndarray:
    assignments = np.asarray(hard_assignment, dtype=np.int64).reshape(-1)
    if not sample_indices:
        raise ValueError("sample_indices must not be empty")
    counts = np.bincount(assignments[np.asarray(sample_indices, dtype=np.int64)], minlength=int(num_slices)).astype(np.float32)
    return _safe_probabilities(counts)


def summarize_mixture(mixture: np.ndarray) -> dict[str, float]:
    probs = _safe_probabilities(mixture)
    safe = np.clip(probs, 1e-12, 1.0)
    return {
        "entropy": float(-(safe * np.log(safe)).sum()),
        "max_mass": float(probs.max()) if probs.size else 0.0,
        "min_mass": float(probs.min()) if probs.size else 0.0,
        "l2_norm": float(np.linalg.norm(probs)) if probs.size else 0.0,
    }


def compute_focus_coverage_stats(
    class_presence: np.ndarray,
    sample_indices: list[int],
    focus_class_indices: Iterable[int] | None = None,
) -> dict[str, object] | None:
    if focus_class_indices is None:
        return None
    matrix = np.asarray(class_presence, dtype=np.int32)
    if matrix.ndim != 2:
        raise ValueError("class_presence must be a 2D array")
    indices = [int(index) for index in focus_class_indices]
    if not indices:
        return None
    subset = matrix[np.asarray(sample_indices, dtype=np.int64)][:, np.asarray(indices, dtype=np.int64)]
    counts = subset.sum(axis=0, dtype=np.int64)
    coverage_ratio = (counts > 0).astype(np.float32)
    return {
        "focus_class_indices": list(indices),
        "focus_class_image_counts": [int(value) for value in counts.tolist()],
        "focus_class_covered_flags": [int(value) for value in coverage_ratio.astype(np.int32).tolist()],
        "focus_class_covered_count": int((counts > 0).sum()),
        "focus_class_total": int(len(indices)),
    }


def flatten_feature_groups(
    feature_groups: dict[str, np.ndarray],
    *,
    feature_label_map: dict[str, list[str]] | None = None,
) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    values: list[np.ndarray] = []
    for block_name in sorted(feature_groups.keys()):
        vector = np.asarray(feature_groups[block_name], dtype=np.float32).reshape(-1)
        labels = list((feature_label_map or {}).get(block_name, []))
        if len(labels) != int(vector.shape[0]):
            labels = [f"dim_{index}" for index in range(int(vector.shape[0]))]
        names.extend(f"{block_name}.{label}" for label in labels)
        values.append(vector)
    if not values:
        return names, np.zeros((0,), dtype=np.float32)
    return names, np.concatenate(values, axis=0).astype(np.float32)


def build_surrogate_feature_payload(
    *,
    feature_groups: dict[str, np.ndarray],
    sample_indices: list[int],
    memberships: np.ndarray,
    hard_assignment: np.ndarray | None = None,
    class_presence: np.ndarray | None = None,
    focus_class_indices: Iterable[int] | None = None,
    feature_label_map: dict[str, list[str]] | None = None,
    include_hard_mixture: bool = False,
) -> dict[str, object]:
    realized_features = aggregate_feature_groups_by_indices(feature_groups, sample_indices)
    raw_names, raw_vector = flatten_feature_groups(
        realized_features,
        feature_label_map=feature_label_map,
    )
    soft_mixture = compute_soft_mixture(memberships, sample_indices)
    mixture_names = [f"mixture.soft.slice_{index:02d}" for index in range(int(soft_mixture.shape[0]))]
    combined_names = list(raw_names) + mixture_names
    combined_vector = np.concatenate([raw_vector, soft_mixture.astype(np.float32)], axis=0)

    payload: dict[str, object] = {
        "sample_count": int(len(sample_indices)),
        "realized_features_raw": serialize_feature_groups(realized_features),
        "realized_features_summary": summarize_feature_groups(realized_features),
        "flat_feature_names": combined_names,
        "flat_feature_vector": [float(value) for value in combined_vector.tolist()],
        "soft_mixture": [float(value) for value in soft_mixture.tolist()],
        "soft_mixture_summary": summarize_mixture(soft_mixture),
    }

    if include_hard_mixture and hard_assignment is not None:
        hard_mixture = compute_hard_mixture(hard_assignment, sample_indices, int(soft_mixture.shape[0]))
        payload["hard_mixture"] = [float(value) for value in hard_mixture.tolist()]
        payload["hard_mixture_summary"] = summarize_mixture(hard_mixture)

    if class_presence is not None:
        focus_stats = compute_focus_coverage_stats(
            class_presence,
            sample_indices,
            focus_class_indices=focus_class_indices,
        )
        if focus_stats is not None:
            payload["focus_coverage"] = focus_stats

    return payload


def build_default_feature_label_map(feature_groups: dict[str, np.ndarray], *, schema_path: str | None = None) -> dict[str, list[str]]:
    return build_feature_label_map(feature_groups, schema_path=schema_path)
