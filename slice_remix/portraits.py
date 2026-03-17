from __future__ import annotations

import os

import numpy as np

try:
    from ..slice_discovery.assembler import ProcessedFeatureAssembler
    from ..slice_discovery.types import ProjectedSliceFeatures
except ImportError:
    from slice_discovery.assembler import ProcessedFeatureAssembler
    from slice_discovery.types import ProjectedSliceFeatures


def compute_slice_portraits(
    feature_groups: dict[str, np.ndarray],
    memberships: np.ndarray,
) -> dict[str, np.ndarray]:
    membership_matrix = np.asarray(memberships, dtype=np.float32)
    if membership_matrix.ndim != 2:
        raise ValueError("memberships must be a 2D array")

    slice_weights = membership_matrix.sum(axis=0, dtype=np.float32)
    safe_weights = np.clip(slice_weights, 1e-12, None)

    portraits: dict[str, np.ndarray] = {}
    for name, matrix in feature_groups.items():
        group_matrix = np.asarray(matrix, dtype=np.float32)
        if group_matrix.ndim == 1:
            group_matrix = group_matrix[:, None]
        if group_matrix.shape[0] != membership_matrix.shape[0]:
            raise ValueError("feature group row count must match memberships")

        portraits[name] = (membership_matrix.T @ group_matrix) / safe_weights[:, None]
    return portraits


def compute_expected_portrait(
    portraits: dict[str, np.ndarray],
    mixture: np.ndarray,
) -> dict[str, np.ndarray]:
    weights = np.asarray(mixture, dtype=np.float32)
    if weights.ndim != 1:
        raise ValueError("mixture must be a 1D array")

    expected: dict[str, np.ndarray] = {}
    for name, portrait_matrix in portraits.items():
        if portrait_matrix.shape[0] != weights.shape[0]:
            raise ValueError("mixture length must match portrait slice dimension")
        expected[name] = weights @ portrait_matrix
    return expected


def compute_portrait_shift(
    portraits: dict[str, np.ndarray],
    baseline_mixture: np.ndarray,
    target_mixture: np.ndarray,
) -> dict[str, np.ndarray]:
    baseline_expected = compute_expected_portrait(portraits, baseline_mixture)
    target_expected = compute_expected_portrait(portraits, target_mixture)
    return {
        name: target_expected[name] - baseline_expected[name]
        for name in portraits
    }


def build_feature_groups_from_projected(projected: ProjectedSliceFeatures) -> dict[str, np.ndarray]:
    return {
        block_name: projected.matrix[:, start:end]
        for block_name, (start, end) in projected.block_ranges.items()
    }


def build_feature_groups_from_assembler(assembler: ProcessedFeatureAssembler) -> dict[str, np.ndarray]:
    return {
        block_name: assembler.get_block(block_name).matrix
        for block_name in assembler.list_blocks()
    }


def load_portrait_feature_groups(
    *,
    projected: ProjectedSliceFeatures,
    cluster_meta: dict[str, object] | None,
    portrait_source: str = "auto",
    processed_data_root: str | None = None,
    schema_path: str | None = None,
) -> tuple[dict[str, np.ndarray], str]:
    if portrait_source not in {"auto", "projected", "semantic"}:
        raise ValueError("portrait_source must be one of: auto, projected, semantic")

    resolved_data_root = processed_data_root or str((cluster_meta or {}).get("data_root", "")).strip() or None
    resolved_schema_path = schema_path or str((cluster_meta or {}).get("schema_path", "")).strip() or None

    if portrait_source in {"auto", "semantic"} and resolved_data_root and resolved_schema_path:
        quality_path = os.path.join(resolved_data_root, "quality", "quality_processed_features.npy")
        difficulty_path = os.path.join(resolved_data_root, "difficulty", "difficulty_processed_features.npy")
        coverage_path = os.path.join(resolved_data_root, "coverage", "coverage_processed_features.npy")
        required_paths = [quality_path, difficulty_path, coverage_path, resolved_schema_path]
        if all(os.path.exists(path) for path in required_paths):
            assembler = ProcessedFeatureAssembler.from_processed_paths(
                quality_path=quality_path,
                difficulty_path=difficulty_path,
                coverage_path=coverage_path,
                schema_path=resolved_schema_path,
            )
            if assembler.sample_ids != projected.sample_ids:
                raise ValueError("processed semantic features must align with projected sample_ids")
            return build_feature_groups_from_assembler(assembler), "semantic"
        if portrait_source == "semantic":
            raise ValueError("semantic portrait source requested but processed bundles could not be resolved")

    if portrait_source == "semantic":
        raise ValueError("semantic portrait source requested but processed_data_root/schema_path are unavailable")
    return build_feature_groups_from_projected(projected), "projected"
