from __future__ import annotations

import json
import os

import numpy as np

from .assembler import ProcessedFeatureAssembler
from .types import ProjectedSliceFeatures


VECTOR_FIELD_NAMES = {"hist", "profile", "delta_profile"}


class SliceFeatureProjector:
    def __init__(
        self,
        scalar_scaler: str = "zscore",
        block_weighting: str = "none",
        pca_components: int | None = None,
    ):
        self.scalar_scaler = scalar_scaler
        self.block_weighting = block_weighting
        self.pca_components = pca_components

    def fit_transform(self, assembler: ProcessedFeatureAssembler) -> ProjectedSliceFeatures:
        matrices: list[np.ndarray] = []
        block_ranges: dict[str, tuple[int, int]] = {}
        cursor = 0

        for block_name in assembler.list_blocks():
            block = assembler.get_block(block_name)
            matrix = block.matrix.astype(np.float32).copy()
            spans = self._field_spans(block.field_names, int(matrix.shape[1]))

            if self.scalar_scaler == "zscore":
                for field_name, start, end in spans:
                    if field_name in VECTOR_FIELD_NAMES or field_name == "empty_flag":
                        continue
                    column = matrix[:, start:end]
                    mean = column.mean(axis=0, keepdims=True)
                    std = column.std(axis=0, keepdims=True)
                    std = np.where(std > 0, std, 1.0)
                    matrix[:, start:end] = (column - mean) / std
            elif self.scalar_scaler != "none":
                raise ValueError(f"Unsupported scalar_scaler='{self.scalar_scaler}'")

            if self.block_weighting == "equal_by_block":
                matrix /= np.sqrt(float(matrix.shape[1]))
            elif self.block_weighting != "none":
                raise ValueError(f"Unsupported block_weighting='{self.block_weighting}'")

            matrices.append(matrix)
            block_ranges[block_name] = (cursor, cursor + int(matrix.shape[1]))
            cursor += int(matrix.shape[1])

        combined = np.concatenate(matrices, axis=1) if matrices else np.zeros((assembler.sample_count, 0), dtype=np.float32)
        return ProjectedSliceFeatures(
            matrix=combined.astype(np.float32),
            sample_ids=list(assembler.sample_ids),
            block_ranges=block_ranges,
        )

    def get_debug_summary(self, projected: ProjectedSliceFeatures) -> dict[str, object]:
        blocks: dict[str, dict[str, object]] = {}
        for name, (start, end) in projected.block_ranges.items():
            block = projected.matrix[:, start:end]
            blocks[name] = {
                "shape": list(block.shape),
                "all_finite": bool(np.isfinite(block).all()),
                "min": float(np.min(block)),
                "max": float(np.max(block)),
                "mean": float(np.mean(block)),
                "std": float(np.std(block)),
            }

        return {
            "sample_count": len(projected.sample_ids),
            "matrix_shape": list(projected.matrix.shape),
            "all_finite": bool(np.isfinite(projected.matrix).all()),
            "block_ranges": {name: [start, end] for name, (start, end) in projected.block_ranges.items()},
            "blocks": blocks,
        }

    def save(self, projected: ProjectedSliceFeatures, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        np.savez(
            os.path.join(output_dir, "projected_features.npz"),
            matrix=projected.matrix,
            sample_ids=np.asarray(projected.sample_ids, dtype=object),
        )
        meta = {
            "sample_ids": list(projected.sample_ids),
            "block_ranges": {name: [start, end] for name, (start, end) in projected.block_ranges.items()},
            "config": {
                "scalar_scaler": self.scalar_scaler,
                "block_weighting": self.block_weighting,
                "pca_components": self.pca_components,
            },
        }
        with open(os.path.join(output_dir, "projected_features_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(output_dir: str) -> ProjectedSliceFeatures:
        payload = np.load(os.path.join(output_dir, "projected_features.npz"), allow_pickle=True)
        with open(os.path.join(output_dir, "projected_features_meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        return ProjectedSliceFeatures(
            matrix=np.asarray(payload["matrix"], dtype=np.float32),
            sample_ids=[str(sample_id) for sample_id in payload["sample_ids"].tolist()],
            block_ranges={name: (int(bounds[0]), int(bounds[1])) for name, bounds in meta["block_ranges"].items()},
        )

    @staticmethod
    def _field_spans(field_names: list[str], width: int) -> list[tuple[str, int, int]]:
        vector_fields = [field_name for field_name in field_names if field_name in VECTOR_FIELD_NAMES]
        if len(vector_fields) > 1:
            raise ValueError("Only one vector field per block is supported in the current projector")

        scalar_count = sum(1 for field_name in field_names if field_name not in VECTOR_FIELD_NAMES)
        vector_width = width - scalar_count
        cursor = 0
        spans: list[tuple[str, int, int]] = []
        for field_name in field_names:
            field_width = vector_width if field_name in VECTOR_FIELD_NAMES else 1
            spans.append((field_name, cursor, cursor + field_width))
            cursor += field_width
        return spans
