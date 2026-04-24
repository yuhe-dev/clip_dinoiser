"""Compatibility bridge to the workspace-level shared dataset feature specs."""

from __future__ import annotations

from pathlib import Path
import sys


_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from feature.features.dataset_specs import (  # noqa: E402,F401
    DatasetFeatureSpec,
    build_thing_id_set,
    get_dataset_feature_spec,
    list_dataset_feature_specs,
    list_dataset_specs_with_num_classes,
    merge_feature_meta_with_dataset_spec,
)

__all__ = [
    "DatasetFeatureSpec",
    "build_thing_id_set",
    "get_dataset_feature_spec",
    "list_dataset_feature_specs",
    "list_dataset_specs_with_num_classes",
    "merge_feature_meta_with_dataset_spec",
]
