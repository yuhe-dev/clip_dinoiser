from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class SliceArtifacts:
    sample_ids: list[str]
    membership: np.ndarray
    hard_assignment: np.ndarray
    slice_weights: np.ndarray
    centers: np.ndarray
    meta: dict[str, object]


def estimate_baseline_mixture(memberships: np.ndarray, sample_indices: list[int]) -> np.ndarray:
    matrix = np.asarray(memberships, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("memberships must be a 2D array")
    if not sample_indices:
        raise ValueError("sample_indices must not be empty")

    selected = matrix[np.asarray(sample_indices, dtype=np.int64)]
    mixture = selected.mean(axis=0, dtype=np.float32)
    if not np.isclose(float(mixture.sum()), 1.0, atol=1e-4):
        raise ValueError("baseline mixture must sum to 1")
    return mixture.astype(np.float32)


def load_slice_artifacts(cluster_dir: str) -> SliceArtifacts:
    payload = np.load(os.path.join(cluster_dir, "slice_result.npz"), allow_pickle=True)
    with open(os.path.join(cluster_dir, "slice_result_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    return SliceArtifacts(
        sample_ids=[str(sample_id) for sample_id in payload["sample_ids"].tolist()],
        membership=np.asarray(payload["membership"], dtype=np.float32),
        hard_assignment=np.asarray(payload["hard_assignment"], dtype=np.int64),
        slice_weights=np.asarray(payload["slice_weights"], dtype=np.float32),
        centers=np.asarray(payload["centers"], dtype=np.float32),
        meta=meta,
    )
