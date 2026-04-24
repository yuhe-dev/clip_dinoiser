from __future__ import annotations

"""Backward-compatible bridge for the VOC feature-prep package.

This module keeps the old import path stable while the implementation now lives
under ``slice_remix.voc_feature_prep`` with clearer module boundaries.
"""

from clip_dinoiser.slice_remix.voc_feature_prep import (
    AVAILABLE_AXIS_NAMES,
    DEFAULT_AXIS_DEFINITIONS,
    DEFAULT_AXIS_NAMES,
    DEFAULT_VOC_ROOT,
    VOC_FOREGROUND_CLASSES,
    VocFeatureAxisDefinition,
    VocFeaturePreparationArtifacts,
    VocTrainAugRecord,
    available_feature_axes,
    build_voc_train_aug_records,
    compute_voc_feature_rows,
    prepare_voc_train_aug_feature_experiment,
    resolve_feature_axes,
)

__all__ = [
    "AVAILABLE_AXIS_NAMES",
    "DEFAULT_AXIS_DEFINITIONS",
    "DEFAULT_AXIS_NAMES",
    "DEFAULT_VOC_ROOT",
    "VOC_FOREGROUND_CLASSES",
    "VocFeatureAxisDefinition",
    "VocFeaturePreparationArtifacts",
    "VocTrainAugRecord",
    "available_feature_axes",
    "build_voc_train_aug_records",
    "compute_voc_feature_rows",
    "prepare_voc_train_aug_feature_experiment",
    "resolve_feature_axes",
]
