from __future__ import annotations

import json
import os
import sys
from importlib import import_module

import numpy as np

from .types import FeatureBlock


def _load_json(path: str) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_numpy_pickle_compat() -> None:
    # Object-array .npy bundles may be pickled against NumPy 2 module paths
    # such as `numpy._core.*`; older runtime environments only expose
    # `numpy.core.*`, so we register aliases before `np.load(..., allow_pickle=True)`.
    try:
        core_module = import_module("numpy._core")
    except ModuleNotFoundError:
        core_module = import_module("numpy.core")
        sys.modules.setdefault("numpy._core", core_module)

    try:
        multiarray_module = import_module("numpy._core.multiarray")
    except ModuleNotFoundError:
        multiarray_module = import_module("numpy.core.multiarray")
        sys.modules.setdefault("numpy._core.multiarray", multiarray_module)


def _debug_print(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _load_records(path: str) -> list[dict[str, object]]:
    _debug_print(f"[load_records] start path={path}")
    _debug_print("[load_records] before compat")
    _ensure_numpy_pickle_compat()
    _debug_print("[load_records] after compat")
    _debug_print("[load_records] before np.load")
    records = np.load(path, allow_pickle=True)
    _debug_print(f"[load_records] after np.load dtype={records.dtype} shape={records.shape}")
    _debug_print("[load_records] before tolist")
    record_list = records.tolist()
    _debug_print(f"[load_records] after tolist len={len(record_list)}")
    dict_records = [dict(record) for record in record_list]
    _debug_print(f"[load_records] after dict cast len={len(dict_records)}")
    return dict_records


_ensure_numpy_pickle_compat()


class ProcessedFeatureAssembler:
    def __init__(
        self,
        sample_ids: list[str],
        blocks: list[FeatureBlock],
        schema_version: str = "",
        dimension_schema_versions: dict[str, str] | None = None,
    ):
        self.sample_ids = list(sample_ids)
        self._blocks = {block.name: block for block in blocks}
        self._block_order = [block.name for block in blocks]
        self.schema_version = schema_version
        self.dimension_schema_versions = dict(dimension_schema_versions or {})

    @property
    def sample_count(self) -> int:
        return len(self.sample_ids)

    @classmethod
    def from_processed_records(
        cls,
        quality_records: list[dict[str, object]],
        difficulty_records: list[dict[str, object]],
        coverage_records: list[dict[str, object]],
        schema: dict[str, object],
    ) -> "ProcessedFeatureAssembler":
        records_by_dimension = {
            "quality": list(quality_records),
            "difficulty": list(difficulty_records),
            "coverage": list(coverage_records),
        }
        cls._validate_alignment(records_by_dimension)
        sample_ids = [str(record["image_rel"]) for record in quality_records]

        blocks: list[FeatureBlock] = []
        dimension_schema_versions = {}
        for dimension_name in ["quality", "difficulty", "coverage"]:
            dimension_schema = dict(schema["dimensions"][dimension_name])
            dimension_schema_versions[dimension_name] = str(dimension_schema.get("schema_version", ""))
            dimension_records = records_by_dimension[dimension_name]
            for feature_name, feature_spec in dimension_schema["features"].items():
                field_names = list(feature_spec["model_input_fields"])
                rows = [
                    cls._extract_feature_row(
                        record=dict(record),
                        feature_name=str(feature_name),
                        field_names=field_names,
                    )
                    for record in dimension_records
                ]
                blocks.append(
                    FeatureBlock(
                        name=f"{dimension_name}.{feature_name}",
                        dimension=dimension_name,
                        feature_name=str(feature_name),
                        field_names=field_names,
                        matrix=np.asarray(rows, dtype=np.float32),
                    )
                )
        return cls(
            sample_ids=sample_ids,
            blocks=blocks,
            schema_version=str(schema.get("schema_version", "")),
            dimension_schema_versions=dimension_schema_versions,
        )

    @classmethod
    def from_processed_paths(
        cls,
        quality_path: str,
        difficulty_path: str,
        coverage_path: str,
        schema_path: str,
        limit_samples: int | None = None,
    ) -> "ProcessedFeatureAssembler":
        quality_records = _load_records(quality_path)
        difficulty_records = _load_records(difficulty_path)
        coverage_records = _load_records(coverage_path)
        if limit_samples is not None:
            quality_records = quality_records[:limit_samples]
            difficulty_records = difficulty_records[:limit_samples]
            coverage_records = coverage_records[:limit_samples]
        return cls.from_processed_records(
            quality_records=quality_records,
            difficulty_records=difficulty_records,
            coverage_records=coverage_records,
            schema=_load_json(schema_path),
        )

    @staticmethod
    def _validate_alignment(records_by_dimension: dict[str, list[dict[str, object]]]) -> None:
        image_ids_by_dimension = {}
        for dimension_name, records in records_by_dimension.items():
            image_ids = [str(record["image_rel"]) for record in records]
            if len(image_ids) != len(set(image_ids)):
                raise ValueError(f"Duplicate image_rel values in {dimension_name} records")
            image_ids_by_dimension[dimension_name] = image_ids

        expected = image_ids_by_dimension["quality"]
        for dimension_name, image_ids in image_ids_by_dimension.items():
            if image_ids != expected:
                raise ValueError(f"Mismatched image_rel ordering for dimension '{dimension_name}'")

    @staticmethod
    def _extract_feature_row(record: dict[str, object], feature_name: str, field_names: list[str]) -> np.ndarray:
        feature = dict(record["features"][feature_name])
        values: list[float] = []
        summary = dict(feature.get("summary", {}))

        for field_name in field_names:
            if field_name in feature:
                raw_value = feature[field_name]
            elif field_name in summary:
                raw_value = summary[field_name]
            else:
                raise KeyError(f"Field '{field_name}' missing for feature '{feature_name}'")

            array = np.asarray(raw_value, dtype=np.float32)
            if array.ndim == 0:
                values.append(float(array))
            else:
                values.extend(array.astype(np.float32).tolist())
        return np.asarray(values, dtype=np.float32)

    def list_blocks(self) -> list[str]:
        return list(self._block_order)

    def get_block(self, name: str) -> FeatureBlock:
        return self._blocks[name]

    def get_flat_view(self, names: list[str] | None = None) -> np.ndarray:
        block_names = list(names) if names is not None else self._block_order
        matrices = [self._blocks[name].matrix for name in block_names]
        if not matrices:
            return np.zeros((self.sample_count, 0), dtype=np.float32)
        return np.concatenate(matrices, axis=1).astype(np.float32)

    def get_metadata(self) -> dict[str, object]:
        block_ranges: dict[str, list[int]] = {}
        cursor = 0
        for name in self._block_order:
            width = int(self._blocks[name].matrix.shape[1])
            block_ranges[name] = [cursor, cursor + width]
            cursor += width

        return {
            "sample_count": self.sample_count,
            "schema_version": self.schema_version,
            "sample_ids": list(self.sample_ids),
            "block_order": list(self._block_order),
            "block_ranges": block_ranges,
            "dimension_schema_versions": dict(self.dimension_schema_versions),
            "blocks": {
                name: {
                    "dimension": self._blocks[name].dimension,
                    "feature_name": self._blocks[name].feature_name,
                    "field_names": list(self._blocks[name].field_names),
                    "shape": list(self._blocks[name].matrix.shape),
                }
                for name in self._block_order
            },
        }

    def get_debug_summary(self) -> dict[str, object]:
        flat = self.get_flat_view()
        blocks: dict[str, dict[str, object]] = {}
        for name in self._block_order:
            matrix = self._blocks[name].matrix
            blocks[name] = {
                "shape": list(matrix.shape),
                "all_finite": bool(np.isfinite(matrix).all()),
                "min": float(np.min(matrix)),
                "max": float(np.max(matrix)),
                "mean": float(np.mean(matrix)),
            }

        return {
            "sample_count": self.sample_count,
            "flat_shape": list(flat.shape),
            "flat_all_finite": bool(np.isfinite(flat).all()),
            "flat_min": float(np.min(flat)) if flat.size else 0.0,
            "flat_max": float(np.max(flat)) if flat.size else 0.0,
            "flat_mean": float(np.mean(flat)) if flat.size else 0.0,
            "blocks": blocks,
        }

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        npz_path = os.path.join(output_dir, "assembled_features.npz")
        meta_path = os.path.join(output_dir, "assembled_features_meta.json")

        payload = {
            "sample_ids": np.asarray(self.sample_ids, dtype=object),
            "X_flat": self.get_flat_view(),
        }
        for name in self._block_order:
            payload[f"block::{name}"] = self._blocks[name].matrix
        np.savez(npz_path, **payload)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.get_metadata(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, output_dir: str) -> "ProcessedFeatureAssembler":
        npz_path = os.path.join(output_dir, "assembled_features.npz")
        meta_path = os.path.join(output_dir, "assembled_features_meta.json")

        matrices = np.load(npz_path, allow_pickle=True)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        blocks: list[FeatureBlock] = []
        for name in metadata["block_order"]:
            block_meta = metadata["blocks"][name]
            blocks.append(
                FeatureBlock(
                    name=name,
                    dimension=block_meta["dimension"],
                    feature_name=block_meta["feature_name"],
                    field_names=list(block_meta["field_names"]),
                    matrix=np.asarray(matrices[f"block::{name}"], dtype=np.float32),
                )
            )

        sample_ids = [str(sample_id) for sample_id in matrices["sample_ids"].tolist()]
        return cls(
            sample_ids=sample_ids,
            blocks=blocks,
            schema_version=str(metadata.get("schema_version", "")),
            dimension_schema_versions=dict(metadata.get("dimension_schema_versions", {})),
        )
