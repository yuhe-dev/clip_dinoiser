from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureBlock:
    name: str
    dimension: str
    feature_name: str
    field_names: list[str]
    matrix: np.ndarray


@dataclass
class ProjectedSliceFeatures:
    matrix: np.ndarray
    sample_ids: list[str]
    block_ranges: dict[str, tuple[int, int]]


@dataclass
class SliceFindingResult:
    sample_ids: list[str]
    membership: np.ndarray
    hard_assignment: np.ndarray
    slice_weights: np.ndarray
    centers: np.ndarray
