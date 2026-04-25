from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


VOC_FOREGROUND_CLASSES: tuple[str, ...] = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


@dataclass(frozen=True)
class VocTrainAugRecord:
    stem: str
    image_rel: str
    annotation_rel: str

    @property
    def basename(self) -> str:
        return os.path.basename(self.image_rel)

    def image_path(self, data_root: str | Path) -> str:
        return str((Path(data_root) / self.image_rel).resolve())

    def annotation_path(self, data_root: str | Path) -> str:
        return str((Path(data_root) / self.annotation_rel).resolve())


@dataclass(frozen=True)
class VocFeatureAxisDefinition:
    key: str
    description: str
    family: str
    enabled_by_default: bool = False


@dataclass(frozen=True)
class VocFeatureComputationResult:
    axis_scores: dict[str, np.ndarray]
    rows: list[dict[str, Any]]
    class_presence_matrix: np.ndarray
    class_presence_rate: np.ndarray
    rarity_weights: np.ndarray
    clipped_rarity_weights: np.ndarray
    foreground_class_count: np.ndarray


@dataclass(frozen=True)
class VocFeaturePreparationArtifacts:
    data_root: str
    subset_size: int
    feature_axes: tuple[str, ...]
    feature_table_path: str
    summary_path: str
    feasibility_report_path: str
    manifest_index_path: str
    manifest_paths: dict[str, str]

    def to_payload(self) -> dict[str, Any]:
        return {
            "data_root": str(self.data_root),
            "subset_size": int(self.subset_size),
            "feature_axes": list(self.feature_axes),
            "feature_table_path": str(self.feature_table_path),
            "summary_path": str(self.summary_path),
            "feasibility_report_path": str(self.feasibility_report_path),
            "manifest_index_path": str(self.manifest_index_path),
            "manifest_paths": dict(self.manifest_paths),
        }
