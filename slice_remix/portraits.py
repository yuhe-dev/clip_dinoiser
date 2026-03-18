from __future__ import annotations

import json
import os

import numpy as np

try:
    from ..slice_discovery.assembler import ProcessedFeatureAssembler
    from ..slice_discovery.types import ProjectedSliceFeatures
except ImportError:
    from slice_discovery.assembler import ProcessedFeatureAssembler
    from slice_discovery.types import ProjectedSliceFeatures


def _noop_log(_: str) -> None:
    return None


VECTOR_FIELD_NAMES = {"hist", "profile", "delta_profile"}


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


def _resolve_assembled_feature_dir(
    *,
    processed_data_root: str | None,
    assembled_feature_dir: str | None,
) -> str | None:
    if assembled_feature_dir:
        return assembled_feature_dir
    if not processed_data_root:
        return None
    candidates = [
        os.path.join(processed_data_root, "assembled_features"),
        os.path.join(processed_data_root, "assembled"),
    ]
    for candidate in candidates:
        npz_path = os.path.join(candidate, "assembled_features.npz")
        meta_path = os.path.join(candidate, "assembled_features_meta.json")
        if os.path.exists(npz_path) and os.path.exists(meta_path):
            return candidate
    return None


def build_feature_label_map(
    feature_groups: dict[str, np.ndarray],
    *,
    schema_path: str | None = None,
) -> dict[str, list[str]]:
    schema = None
    if schema_path and os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

    labels: dict[str, list[str]] = {}
    for block_name, matrix in feature_groups.items():
        width = int(np.asarray(matrix).shape[1]) if np.asarray(matrix).ndim == 2 else 1
        if schema is None or "." not in block_name:
            labels[block_name] = [f"dim_{index}" for index in range(width)]
            continue

        dimension_name, feature_name = block_name.split(".", 1)
        feature_spec = (
            schema.get("dimensions", {})
            .get(dimension_name, {})
            .get("features", {})
            .get(feature_name, {})
        )
        field_names = list(feature_spec.get("model_input_fields", []))
        if not field_names:
            labels[block_name] = [f"dim_{index}" for index in range(width)]
            continue

        scalar_count = sum(1 for field_name in field_names if field_name not in VECTOR_FIELD_NAMES)
        vector_width = max(width - scalar_count, 0)
        block_labels: list[str] = []
        for field_name in field_names:
            if field_name in VECTOR_FIELD_NAMES:
                block_labels.extend(f"{field_name}[{index}]" for index in range(vector_width))
            else:
                block_labels.append(str(field_name))
        if len(block_labels) != width:
            labels[block_name] = [f"dim_{index}" for index in range(width)]
            continue
        labels[block_name] = block_labels
    return labels


def summarize_portrait_shift(
    delta_phi: dict[str, list[float] | np.ndarray],
    feature_label_map: dict[str, list[str]],
    *,
    top_blocks: int = 5,
    top_features_per_block: int = 3,
) -> dict[str, object]:
    block_summaries: list[dict[str, object]] = []
    for block_name, raw_values in delta_phi.items():
        values = np.asarray(raw_values, dtype=np.float32).reshape(-1)
        labels = feature_label_map.get(block_name, [f"dim_{index}" for index in range(values.shape[0])])
        abs_values = np.abs(values)
        order = np.argsort(-abs_values)
        top_feature_rows = [
            {
                "feature": labels[int(index)] if int(index) < len(labels) else f"dim_{int(index)}",
                "index": int(index),
                "delta": float(values[int(index)]),
                "abs_delta": float(abs_values[int(index)]),
            }
            for index in order[:top_features_per_block]
        ]
        block_summaries.append(
            {
                "block_name": block_name,
                "dimension": int(values.shape[0]),
                "l2_norm": float(np.linalg.norm(values)),
                "mean_abs_delta": float(abs_values.mean()) if abs_values.size else 0.0,
                "max_abs_delta": float(abs_values.max()) if abs_values.size else 0.0,
                "signed_sum": float(values.sum()) if values.size else 0.0,
                "top_features": top_feature_rows,
            }
        )

    block_summaries.sort(key=lambda item: float(item["l2_norm"]), reverse=True)
    return {
        "top_blocks": block_summaries[:top_blocks],
        "num_blocks": len(block_summaries),
    }


def load_portrait_feature_groups(
    *,
    projected: ProjectedSliceFeatures,
    cluster_meta: dict[str, object] | None,
    portrait_source: str = "auto",
    processed_data_root: str | None = None,
    schema_path: str | None = None,
    assembled_feature_dir: str | None = None,
    log_fn=None,
) -> tuple[dict[str, np.ndarray], str]:
    if portrait_source not in {"auto", "projected", "semantic"}:
        raise ValueError("portrait_source must be one of: auto, projected, semantic")

    logger = log_fn or _noop_log
    resolved_data_root = processed_data_root or str((cluster_meta or {}).get("data_root", "")).strip() or None
    resolved_schema_path = schema_path or str((cluster_meta or {}).get("schema_path", "")).strip() or None
    resolved_assembled_dir = _resolve_assembled_feature_dir(
        processed_data_root=resolved_data_root,
        assembled_feature_dir=assembled_feature_dir,
    )

    if portrait_source in {"auto", "semantic"} and resolved_data_root and resolved_schema_path:
        if resolved_assembled_dir:
            logger(f"loading semantic assembled features dir={resolved_assembled_dir}")
            assembler = ProcessedFeatureAssembler.load(resolved_assembled_dir)
            logger(f"loaded semantic assembled features dir={resolved_assembled_dir} sample_count={assembler.sample_count}")
            if assembler.sample_ids != projected.sample_ids:
                raise ValueError("assembled semantic features must align with projected sample_ids")
            return build_feature_groups_from_assembler(assembler), "semantic"

        quality_path = os.path.join(resolved_data_root, "quality", "quality_processed_features.npy")
        difficulty_path = os.path.join(resolved_data_root, "difficulty", "difficulty_processed_features.npy")
        coverage_path = os.path.join(resolved_data_root, "coverage", "coverage_processed_features.npy")
        required_paths = [quality_path, difficulty_path, coverage_path, resolved_schema_path]
        if all(os.path.exists(path) for path in required_paths):
            logger(f"loading semantic processed features data_root={resolved_data_root}")
            assembler = ProcessedFeatureAssembler.from_processed_paths(
                quality_path=quality_path,
                difficulty_path=difficulty_path,
                coverage_path=coverage_path,
                schema_path=resolved_schema_path,
                log_fn=logger,
            )
            logger(f"assembled semantic processed features sample_count={assembler.sample_count} blocks={len(assembler.list_blocks())}")
            if assembler.sample_ids != projected.sample_ids:
                raise ValueError("processed semantic features must align with projected sample_ids")
            return build_feature_groups_from_assembler(assembler), "semantic"
        if portrait_source == "semantic":
            raise ValueError("semantic portrait source requested but processed bundles could not be resolved")

    if portrait_source == "semantic":
        raise ValueError("semantic portrait source requested but processed_data_root/schema_path are unavailable")
    return build_feature_groups_from_projected(projected), "projected"
