from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

if __package__ in {None, ""}:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from slice_discovery.projector import SliceFeatureProjector
    from slice_remix.baseline import estimate_baseline_mixture, load_slice_artifacts
    from slice_remix.portraits import (
        build_feature_label_map,
        compute_slice_portraits,
        load_portrait_feature_groups,
    )
    from slice_remix.prior_graph import PriorGraphHyperparams, build_prior_graph
else:
    from .slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from .slice_discovery.projector import SliceFeatureProjector
    from .slice_remix.baseline import estimate_baseline_mixture, load_slice_artifacts
    from .slice_remix.portraits import (
        build_feature_label_map,
        compute_slice_portraits,
        load_portrait_feature_groups,
    )
    from .slice_remix.prior_graph import PriorGraphHyperparams, build_prior_graph


DIMENSION_ORDER = ("quality", "difficulty", "coverage")
EXCLUDED_MODEL_FIELDS = {"log_num_values", "empty_flag"}
VECTOR_FIELD_NAMES = {"hist", "profile", "delta_profile"}
DEFAULT_METRIC_PATH = "coco_stuff.summary.mIoU"
PREFERRED_SUMMARY_FIELDS = {
    "quality.laplacian": ("q50", "q90", "low_sharpness_mass"),
    "quality.noise_pca": ("q50", "q90", "high_noise_mass"),
    "quality.bga": ("q50", "q10", "high_bga_mass"),
    "difficulty.small_ratio": ("mass_small_extreme", "mass_small_mid"),
    "difficulty.visual_semantic_gap": ("q50", "q90", "high_gap_mass"),
    "difficulty.empirical_iou": ("q50", "q10", "low_iou_mass"),
    "coverage.knn_local_density": ("density_score", "q50", "nearest_distance"),
    "coverage.prototype_distance": ("nearest_prototype_distance", "q50", "prototype_margin_top2"),
}


def _progress(message: str) -> None:
    print(f"[workbench_bundle_export] {message}", file=sys.stderr, flush=True)


def _load_json(path: str | Path) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str | Path, payload: object) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _infer_sample_url_prefix(slice_report_dir: str | Path) -> str:
    resolved = Path(slice_report_dir).resolve()
    parts = list(resolved.parts)
    if "public" not in parts:
        return ""
    public_index = parts.index("public")
    suffix = "/".join(parts[public_index + 1 :])
    if not suffix:
        return ""
    return f"/{suffix}"


def _humanize(token: str) -> str:
    replacements = {
        "q10": "Q10",
        "q25": "Q25",
        "q50": "Q50",
        "q75": "Q75",
        "q90": "Q90",
        "bga": "BGA",
        "knn": "KNN",
        "iou": "IoU",
        "pca": "PCA",
        "delta_profile": "Delta Profile",
        "profile": "Profile",
        "hist": "Histogram",
    }
    parts = token.split("_")
    labels = []
    for part in parts:
        if part in replacements:
            labels.append(replacements[part])
        elif part:
            labels.append(part[0].upper() + part[1:])
    return " ".join(labels)


def _field_label(field_name: str) -> str:
    if field_name == "hist":
        return "Histogram"
    if field_name == "profile":
        return "Profile"
    if field_name == "delta_profile":
        return "Delta Profile"
    return _humanize(field_name)


def _metric_value_from_path(payload: dict[str, object], metric_path: str) -> float | None:
    current: object = payload
    for token in metric_path.split("."):
        if not isinstance(current, dict) or token not in current:
            return None
        current = current[token]
    if current is None:
        return None
    return float(current)


def _safe_normalize(weights: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    total = float(np.sum(weights, dtype=np.float64))
    if total <= epsilon:
        return np.zeros_like(weights, dtype=np.float32)
    return (weights / total).astype(np.float32)


def _uniform_sample_weights(num_samples: int) -> np.ndarray:
    return np.full((num_samples,), 1.0 / float(num_samples), dtype=np.float32)


def _selected_sample_weights(num_samples: int, sample_indices: list[int]) -> np.ndarray:
    weights = np.zeros((num_samples,), dtype=np.float32)
    if not sample_indices:
        return weights
    weights[np.asarray(sample_indices, dtype=np.int64)] = 1.0
    return _safe_normalize(weights)


def _mixture_to_sample_weights(mixture: np.ndarray, memberships: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    membership_matrix = np.asarray(memberships, dtype=np.float32)
    safe_slice_mass = np.clip(membership_matrix.sum(axis=0, dtype=np.float32), epsilon, None)
    weights = membership_matrix @ (np.asarray(mixture, dtype=np.float32) / safe_slice_mass)
    return _safe_normalize(weights, epsilon=epsilon)


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles: Iterable[float], epsilon: float = 1e-8) -> list[float]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    weight_vector = np.asarray(weights, dtype=np.float64).reshape(-1)
    positive = weight_vector > epsilon
    if not np.any(positive):
        return [0.0 for _ in quantiles]
    vector = vector[positive]
    weight_vector = weight_vector[positive]
    order = np.argsort(vector, kind="mergesort")
    sorted_values = vector[order]
    sorted_weights = weight_vector[order]
    cumulative = np.cumsum(sorted_weights)
    total = float(cumulative[-1])
    targets = np.asarray(list(quantiles), dtype=np.float64) * total
    return np.interp(targets, cumulative, sorted_values).astype(float).tolist()


def _build_scalar_summary(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    normalized = _safe_normalize(np.asarray(weights, dtype=np.float32))
    mean = float(np.sum(np.asarray(values, dtype=np.float64) * normalized.astype(np.float64)))
    q25, q50, q75, q90 = _weighted_quantiles(values, normalized, [0.25, 0.50, 0.75, 0.90])
    return {
        "mean": mean,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "q90": q90,
    }


def _build_vector_series(matrix: np.ndarray, weights: np.ndarray) -> dict[str, object]:
    normalized = _safe_normalize(np.asarray(weights, dtype=np.float32))
    values = np.asarray(matrix, dtype=np.float32)
    vector = np.sum(normalized[:, None] * values, axis=0, dtype=np.float32)
    count = int(vector.shape[0])
    bins = np.linspace(1.0 / float(count), 1.0, count, dtype=np.float32)
    return {
        "bins": [float(value) for value in bins.tolist()],
        "values": [float(value) for value in vector.tolist()],
    }


def _resolve_baseline_sample_indices(
    *,
    manifest_path: str | None,
    fallback_seed: int,
    fallback_budget: int,
    sample_ids: list[str],
) -> tuple[list[int], str]:
    sample_index = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    if manifest_path:
        manifest = _load_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError("baseline manifest must be a JSON object")
        selected_ids = [str(sample_id) for sample_id in manifest.get("sample_ids", [])]
        indices = [sample_index[sample_id] for sample_id in selected_ids if sample_id in sample_index]
        if not indices:
            raise ValueError(f"baseline manifest {manifest_path} did not resolve to any atlas sample ids")
        return indices, os.path.abspath(manifest_path)

    rng = np.random.default_rng(int(fallback_seed))
    indices = rng.choice(len(sample_ids), size=int(fallback_budget), replace=False).tolist()
    return indices, "rng_fallback"


def _iter_recommendation_candidates(bundle_root: str, round_count: int, baseline_mixture: np.ndarray) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {}
    for round_id in range(1, int(round_count) + 1):
        path = Path(bundle_root) / f"recommendation_round_{round_id}.json"
        if not path.exists():
            continue
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        for row in payload.get("candidate_rankings", []):
            candidate_id = str(row.get("candidate_id", "")).strip()
            delta_q = np.asarray(row.get("delta_q", []), dtype=np.float32)
            if not candidate_id or delta_q.shape != baseline_mixture.shape:
                continue
            mixture = np.asarray(baseline_mixture + delta_q, dtype=np.float32)
            mixture = np.clip(mixture, 0.0, None)
            total = float(mixture.sum())
            if total > 0:
                mixture /= total
            candidates[candidate_id] = mixture.astype(np.float32)
    return candidates


def _resolve_candidate_realized_weights(
    *,
    candidate_ids: Iterable[str],
    candidate_manifest_dirs: list[str],
    sample_ids: list[str],
) -> dict[str, np.ndarray]:
    sample_index = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    realized: dict[str, np.ndarray] = {}
    for candidate_id in candidate_ids:
        resolved_manifest: Path | None = None
        for manifest_dir in candidate_manifest_dirs:
            candidate_path = Path(manifest_dir) / f"{candidate_id}.json"
            if candidate_path.exists():
                resolved_manifest = candidate_path
                break
        if resolved_manifest is None:
            continue
        payload = _load_json(resolved_manifest)
        if not isinstance(payload, dict):
            continue
        selected_ids = [str(sample_id) for sample_id in payload.get("sample_ids", [])]
        indices = [sample_index[sample_id] for sample_id in selected_ids if sample_id in sample_index]
        if indices:
            realized[candidate_id] = _selected_sample_weights(len(sample_ids), indices)
    return realized


def _block_to_dimension(block_name: str) -> str:
    return block_name.split(".", 1)[0] if "." in block_name else "quality"


def _group_indices_for_field(labels: list[str], field_name: str) -> list[int]:
    prefix = f"{field_name}["
    if field_name in VECTOR_FIELD_NAMES:
        return [idx for idx, label in enumerate(labels) if label.startswith(prefix)]
    return [idx for idx, label in enumerate(labels) if label == field_name]


def _distribution_type(field_name: str) -> str:
    if field_name == "hist":
        return "histogram"
    if field_name in {"profile", "delta_profile"}:
        return "step_curve"
    return "scalar_interval"


def _build_field_distributions(
    *,
    schema: dict[str, object],
    feature_groups: dict[str, np.ndarray],
    feature_label_map: dict[str, list[str]],
    sample_weight_sets: dict[str, np.ndarray],
    slice_weight_sets: dict[str, np.ndarray],
    candidate_expected_weight_sets: dict[str, np.ndarray],
    candidate_realized_weight_sets: dict[str, np.ndarray],
) -> dict[str, object]:
    blocks_payload: list[dict[str, object]] = []
    for dimension_name in DIMENSION_ORDER:
        dimension_spec = schema["dimensions"][dimension_name]
        feature_rows: list[dict[str, object]] = []
        for feature_name, feature_spec in dimension_spec["features"].items():
            block_name = f"{dimension_name}.{feature_name}"
            matrix = np.asarray(feature_groups[block_name], dtype=np.float32)
            labels = feature_label_map[block_name]

            field_rows: list[dict[str, object]] = []
            for model_field in feature_spec["model_input_fields"]:
                if model_field in EXCLUDED_MODEL_FIELDS:
                    continue
                column_indices = _group_indices_for_field(labels, model_field)
                if not column_indices:
                    continue
                field_matrix = matrix[:, np.asarray(column_indices, dtype=np.int64)]
                distribution_type = _distribution_type(model_field)

                if distribution_type == "scalar_interval":
                    values = field_matrix[:, 0]
                    global_pool = {"summary": _build_scalar_summary(values, sample_weight_sets["global_pool"])}
                    baseline = {"summary": _build_scalar_summary(values, sample_weight_sets["baseline"])}
                    canonical_slices = {
                        slice_id: {"summary": _build_scalar_summary(values, weights)}
                        for slice_id, weights in slice_weight_sets.items()
                    }
                    candidate_expected = {
                        candidate_id: {"summary": _build_scalar_summary(values, weights)}
                        for candidate_id, weights in candidate_expected_weight_sets.items()
                    }
                    candidate_realized = {
                        candidate_id: {"summary": _build_scalar_summary(values, weights)}
                        for candidate_id, weights in candidate_realized_weight_sets.items()
                    }
                else:
                    global_pool = _build_vector_series(field_matrix, sample_weight_sets["global_pool"])
                    baseline = _build_vector_series(field_matrix, sample_weight_sets["baseline"])
                    canonical_slices = {
                        slice_id: _build_vector_series(field_matrix, weights)
                        for slice_id, weights in slice_weight_sets.items()
                    }
                    candidate_expected = {
                        candidate_id: _build_vector_series(field_matrix, weights)
                        for candidate_id, weights in candidate_expected_weight_sets.items()
                    }
                    candidate_realized = {
                        candidate_id: _build_vector_series(field_matrix, weights)
                        for candidate_id, weights in candidate_realized_weight_sets.items()
                    }

                field_rows.append(
                    {
                        "feature_name": feature_name,
                        "feature_label": _humanize(feature_name),
                        "field_name": f"{feature_name}_{model_field}",
                        "field_label": _field_label(model_field),
                        "distribution_type": distribution_type,
                        "global_pool": global_pool,
                        "baseline": baseline,
                        "canonical_slices": canonical_slices,
                        "candidate_expected": candidate_expected,
                        "candidate_realized": candidate_realized,
                    }
                )

            feature_rows.append(
                {
                    "feature_name": feature_name,
                    "feature_label": _humanize(feature_name),
                    "fields": field_rows,
                }
            )

        blocks_payload.append(
            {
                "block_name": dimension_name,
                "block_label": _humanize(dimension_name),
                "features": feature_rows,
            }
        )

    return {"blocks": blocks_payload}


def _compute_summary_proxy_values(
    *,
    schema: dict[str, object],
    feature_groups: dict[str, np.ndarray],
    feature_label_map: dict[str, list[str]],
    slice_portraits: dict[str, np.ndarray],
    memberships: np.ndarray,
    baseline_weights: np.ndarray,
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, float]]:
    proxy_by_slice: dict[str, dict[str, float]] = {}
    proxy_pool: dict[str, float] = {}
    proxy_baseline: dict[str, float] = {}
    pool_weights = _uniform_sample_weights(memberships.shape[0])

    for dimension_name in DIMENSION_ORDER:
        for feature_name, _feature_spec in schema["dimensions"][dimension_name]["features"].items():
            block_name = f"{dimension_name}.{feature_name}"
            preferred = PREFERRED_SUMMARY_FIELDS.get(block_name, ())
            labels = feature_label_map[block_name]
            chosen_index = None
            for candidate in preferred:
                if candidate in labels:
                    chosen_index = labels.index(candidate)
                    break
            if chosen_index is None:
                scalar_candidates = [
                    idx for idx, label in enumerate(labels) if not any(label.startswith(f"{prefix}[") for prefix in VECTOR_FIELD_NAMES)
                ]
                chosen_index = scalar_candidates[0] if scalar_candidates else 0

            sample_values = np.asarray(feature_groups[block_name][:, chosen_index], dtype=np.float32)
            low, high = np.quantile(sample_values, [0.05, 0.95])
            if float(high - low) < 1e-6:
                low = float(np.min(sample_values))
                high = float(np.max(sample_values) + 1e-6)

            def normalize(value: float) -> float:
                return float(np.clip((value - low) / (high - low), 0.0, 1.0))

            proxy_pool[block_name] = normalize(float(np.sum(sample_values * pool_weights)))
            proxy_baseline[block_name] = normalize(float(np.sum(sample_values * baseline_weights)))
            for slice_index in range(slice_portraits[block_name].shape[0]):
                proxy_by_slice.setdefault(block_name, {})
                proxy_by_slice[block_name][f"slice_{slice_index:02d}"] = normalize(float(slice_portraits[block_name][slice_index, chosen_index]))

    return proxy_by_slice, proxy_pool, proxy_baseline


def _dimension_summary_from_proxy(
    proxy_by_slice: dict[str, dict[str, float]],
    *,
    slice_id: str,
) -> dict[str, list[float]]:
    summary: dict[str, list[float]] = {dimension: [] for dimension in DIMENSION_ORDER}
    for block_name, slice_values in proxy_by_slice.items():
        dimension = _block_to_dimension(block_name)
        summary[dimension].append(float(slice_values[slice_id]))
    return summary


def _build_slice_atlas(
    *,
    slice_report_slices: list[dict[str, object]],
    prior_graph_payload: dict[str, object],
    proxy_by_slice: dict[str, dict[str, float]],
) -> dict[str, object]:
    node_by_id = {node["slice_id"]: node for node in prior_graph_payload["nodes"]}
    slice_rows: list[dict[str, object]] = []
    for row in slice_report_slices:
        slice_id = str(row["slice_id"])
        node = node_by_id[slice_id]
        summary = _dimension_summary_from_proxy(proxy_by_slice, slice_id=slice_id)
        shifted_tags: list[str] = []
        for feature in row.get("top_shifted_features", []):
            block_name = str(feature.get("block", ""))
            feature_tag = _humanize(block_name.split(".", 1)[1]) if "." in block_name else _humanize(block_name)
            if feature_tag and feature_tag not in shifted_tags:
                shifted_tags.append(feature_tag)
            if len(shifted_tags) >= 2:
                break

        semantic_tags = []
        semantic_tags.append("Underrepresented" if float(node["pool_delta"]) > 0.02 else "Overrepresented" if float(node["pool_delta"]) < -0.02 else "Balanced")
        semantic_tags.append("Stable" if float(node["instability_score"]) < 0.45 else "Diffuse")
        semantic_tags.extend(shifted_tags)

        slice_rows.append(
            {
                "slice_id": slice_id,
                "index": int(row["index"]),
                "canonical_weight": float(node["canonical_weight"]),
                "baseline_weight": float(node["baseline_weight"]),
                "semantic_tags": semantic_tags[:3],
                "quality_summary": [float(value) for value in summary["quality"]],
                "difficulty_summary": [float(value) for value in summary["difficulty"]],
                "coverage_summary": [float(value) for value in summary["coverage"]],
                "representative_sample_ids": [str(sample_id) for sample_id in row.get("representative_samples", [])[:6]],
                "boundary_sample_ids": [str(sample_id) for sample_id in row.get("ambiguous_samples", [])[:6]],
            }
        )

    return {"slices": slice_rows}


def _build_baseline_footprint(
    *,
    baseline_seed: int,
    budget: int,
    baseline_mixture: np.ndarray,
    pool_mixture: np.ndarray,
    proxy_by_slice: dict[str, dict[str, float]],
    proxy_pool: dict[str, float],
    proxy_baseline: dict[str, float],
    baseline_result_payload: dict[str, object] | None,
    metric_path: str,
) -> dict[str, object]:
    delta = pool_mixture - baseline_mixture
    over_indices = np.argsort(-(baseline_mixture - pool_mixture))
    under_indices = np.argsort(-(pool_mixture - baseline_mixture))
    baseline_dimension_delta: dict[str, float] = defaultdict(float)
    pool_dimension_delta: dict[str, float] = defaultdict(float)
    per_slice_rows = []
    for slice_index in range(len(baseline_mixture)):
        slice_id = f"slice_{slice_index:02d}"
        dimension_rows: dict[str, float] = {}
        for dimension_name in DIMENSION_ORDER:
            block_names = [block_name for block_name in proxy_by_slice if _block_to_dimension(block_name) == dimension_name]
            slice_values = [proxy_by_slice[block_name][slice_id] for block_name in block_names]
            pool_values = [proxy_pool[block_name] for block_name in block_names]
            dimension_rows[dimension_name] = float(np.mean(slice_values) - np.mean(pool_values)) if block_names else 0.0
        per_slice_rows.append(
            {
                "slice_id": slice_id,
                "baseline_quality_delta": dimension_rows["quality"],
                "baseline_difficulty_delta": dimension_rows["difficulty"],
                "baseline_coverage_delta": dimension_rows["coverage"],
            }
        )

    for block_name, baseline_value in proxy_baseline.items():
        dimension = _block_to_dimension(block_name)
        baseline_dimension_delta[dimension] += baseline_value
        pool_dimension_delta[dimension] += proxy_pool[block_name]

    dimension_gap = {
        dimension: baseline_dimension_delta[dimension] - pool_dimension_delta[dimension]
        for dimension in DIMENSION_ORDER
    }
    dominant_dimension = max(dimension_gap, key=lambda key: abs(dimension_gap[key]))
    distortion_tags = [
        "Underrepresented slices: " + ", ".join(f"slice_{idx:02d}" for idx in under_indices[:3]),
        "Overrepresented slices: " + ", ".join(f"slice_{idx:02d}" for idx in over_indices[:3]),
        f"Largest baseline portrait drift is in {dominant_dimension}.",
    ]

    metric_value = _metric_value_from_path(baseline_result_payload or {}, metric_path)
    return {
        "baseline_id": f"baseline_seed{baseline_seed}_b{budget}",
        "mixture": [float(value) for value in baseline_mixture.tolist()],
        "pool_vs_baseline_delta": [float(value) for value in delta.tolist()],
        "top_overrepresented_slices": [f"slice_{index:02d}" for index in over_indices[:3]],
        "top_underrepresented_slices": [f"slice_{index:02d}" for index in under_indices[:3]],
        "distortion_tags": distortion_tags,
        "baseline_performance": {
            "metric_name": metric_path,
            "value": float(metric_value) if metric_value is not None else 0.0,
        },
        "slice_realizations": per_slice_rows,
    }


def _rewrite_sample_url(image_url: str, sample_url_prefix: str) -> str:
    if image_url.startswith("./thumbnails/") and sample_url_prefix:
        return f"{sample_url_prefix}/{image_url[2:]}"
    return image_url


def _build_samples_payload(
    *,
    slice_report_samples: list[dict[str, object]],
    slice_atlas_payload: dict[str, object],
    sample_url_prefix: str,
) -> dict[str, object]:
    sample_ids_needed = set()
    sample_roles: dict[str, list[str]] = defaultdict(list)
    linked_blocks: dict[str, set[str]] = defaultdict(set)

    for slice_row in slice_atlas_payload["slices"]:
        slice_id = slice_row["slice_id"]
        for sample_id in slice_row["representative_sample_ids"]:
            sample_ids_needed.add(sample_id)
            sample_roles[sample_id].append(f"Representative for {slice_id}")
            linked_blocks[sample_id].update(["quality", "difficulty", "coverage"])
        for sample_id in slice_row["boundary_sample_ids"]:
            sample_ids_needed.add(sample_id)
            sample_roles[sample_id].append(f"Ambiguous sample near {slice_id}")
            linked_blocks[sample_id].update(["difficulty", "coverage"])

    samples_by_id = {str(sample["sample_id"]): sample for sample in slice_report_samples}
    payload_rows: list[dict[str, object]] = []
    for sample_id in sorted(sample_ids_needed):
        source = samples_by_id.get(sample_id)
        if source is None:
            continue
        image_url = _rewrite_sample_url(str(source.get("image_url", "")), sample_url_prefix)
        payload_rows.append(
            {
                "sample_id": sample_id,
                "image_url": image_url,
                "thumbnail_url": image_url,
                "hard_assignment": int(source.get("hard_assignment", 0)),
                "membership_vector": [float(value) for value in source.get("membership_vector", [])],
                "slice_rankings": [int(value) for value in source.get("slice_rankings", [])[:4]],
                "sample_notes": " · ".join(sample_roles.get(sample_id, [])),
                "linked_blocks": sorted(linked_blocks.get(sample_id, set())),
            }
        )
    return {"samples": payload_rows}


def _build_slice_relationships(
    *,
    slice_ids: list[str],
    prior_graph_payload: dict[str, object],
    slice_report_centers_2d: list[dict[str, object]],
    slice_portraits: dict[str, np.ndarray],
) -> dict[str, object]:
    edge_score = {(edge["donor"], edge["receiver"]): float(edge["score"]) for edge in prior_graph_payload["edges"]}
    visible_scores = [value for value in edge_score.values()]
    score_min = min(visible_scores) if visible_scores else 0.0
    score_max = max(visible_scores) if visible_scores else 1.0

    def normalize_score(value: float) -> float:
        if score_max - score_min < 1e-8:
            return 0.0
        return (value - score_min) / (score_max - score_min)

    portrait_blocks = [np.asarray(slice_portraits[name], dtype=np.float32) for name in sorted(slice_portraits)]
    portrait_matrix = np.concatenate(portrait_blocks, axis=1) if portrait_blocks else np.zeros((len(slice_ids), 0), dtype=np.float32)
    normalized = portrait_matrix.copy()
    if normalized.size:
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        normalized = normalized / norms
    similarity = normalized @ normalized.T if normalized.size else np.eye(len(slice_ids), dtype=np.float32)
    similarity = np.clip(similarity, -1.0, 1.0)
    distance = 1.0 - similarity

    transfer = []
    ambiguity_pairs = []
    for donor in slice_ids:
        row = []
        for receiver in slice_ids:
            if donor == receiver:
                row.append(0.0)
                continue
            row.append(float(normalize_score(edge_score.get((donor, receiver), 0.0))))
        transfer.append(row)

    upper_pairs = []
    for i, left in enumerate(slice_ids):
        for j in range(i + 1, len(slice_ids)):
            right = slice_ids[j]
            upper_pairs.append((float(similarity[i, j]), left, right))
    upper_pairs.sort(reverse=True)
    for score, left, right in upper_pairs[:6]:
        ambiguity_pairs.append({"left": left, "right": right, "score": score})

    point_rows = [
        {
            "slice_id": str(row["slice_id"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
        }
        for row in slice_report_centers_2d
    ]

    return {
        "slice_ids": slice_ids,
        "similarity_matrix": [[float(value) for value in row] for row in similarity.tolist()],
        "distance_matrix": [[float(value) for value in row] for row in distance.tolist()],
        "transfer_potential_matrix": [[float(value) for value in row] for row in transfer],
        "ambiguity_pairs": ambiguity_pairs,
        "projection_points": point_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a real backend-derived workbench bundle for SliceTune.")
    parser.add_argument("--projected-dir", required=True)
    parser.add_argument("--cluster-dir", required=True)
    parser.add_argument("--slice-report-dir", required=True)
    parser.add_argument("--processed-data-root")
    parser.add_argument("--schema-path", required=True)
    parser.add_argument("--input-bundle-root", required=True)
    parser.add_argument("--output-root", action="append", required=True)
    parser.add_argument("--portrait-source", choices=["auto", "projected", "semantic"], default="semantic")
    parser.add_argument("--baseline-seed", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--baseline-manifest-path")
    parser.add_argument("--baseline-result-path")
    parser.add_argument("--candidate-manifest-dir", action="append", default=[])
    parser.add_argument("--metric-path", default=DEFAULT_METRIC_PATH)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--top-k-render", type=int, default=12)
    return parser


def run(args: argparse.Namespace, log_fn=_progress) -> int:
    input_bundle_root = Path(args.input_bundle_root).resolve()
    task_context = _load_json(input_bundle_root / "task_context.json")
    if not isinstance(task_context, dict):
        raise ValueError("task_context.json must be a JSON object")

    baseline_seed = int(args.baseline_seed if args.baseline_seed is not None else task_context.get("baseline_seed", 0))
    budget = int(args.budget if args.budget is not None else task_context.get("baseline_budget", 0))
    round_count = int(task_context.get("round_count", 3))

    projected = SliceFeatureProjector.load(os.path.abspath(args.projected_dir))
    artifacts = load_slice_artifacts(os.path.abspath(args.cluster_dir))
    if projected.sample_ids != artifacts.sample_ids:
        raise ValueError("projected sample ids must match cluster sample ids")

    feature_groups, portrait_source = load_portrait_feature_groups(
        projected=projected,
        cluster_meta=artifacts.meta,
        portrait_source=args.portrait_source,
        processed_data_root=os.path.abspath(args.processed_data_root) if args.processed_data_root else None,
        schema_path=os.path.abspath(args.schema_path),
        log_fn=log_fn,
    )
    feature_label_map = build_feature_label_map(
        feature_groups,
        schema_path=os.path.abspath(args.schema_path),
    )
    schema = _load_json(args.schema_path)
    if not isinstance(schema, dict):
        raise ValueError("schema must be a JSON object")

    baseline_sample_indices, baseline_source = _resolve_baseline_sample_indices(
        manifest_path=args.baseline_manifest_path,
        fallback_seed=baseline_seed,
        fallback_budget=budget,
        sample_ids=artifacts.sample_ids,
    )
    log_fn(f"baseline sample source={baseline_source} size={len(baseline_sample_indices)}")

    baseline_mixture = estimate_baseline_mixture(artifacts.membership, baseline_sample_indices)
    pool_mixture = artifacts.membership.mean(axis=0, dtype=np.float32)
    slice_ids = [f"slice_{index:02d}" for index in range(artifacts.membership.shape[1])]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        slice_portraits = compute_slice_portraits(feature_groups, artifacts.membership)

        prior_graph = build_prior_graph(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=artifacts.membership,
            baseline_sample_indices=baseline_sample_indices,
            slice_ids=slice_ids,
            hyperparams=PriorGraphHyperparams(
                top_k_render=int(args.top_k_render),
                score_threshold=float(args.score_threshold),
            ),
            baseline_seed=baseline_seed,
            budget=budget,
        ).to_dict()
    prior_graph["graph_context"]["portrait_source"] = portrait_source
    prior_graph["graph_context"]["baseline_manifest_path"] = baseline_source

    sample_weight_sets = {
        "global_pool": _uniform_sample_weights(len(artifacts.sample_ids)),
        "baseline": _selected_sample_weights(len(artifacts.sample_ids), baseline_sample_indices),
    }
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        slice_weight_sets = {
            slice_id: _mixture_to_sample_weights(
                np.eye(len(slice_ids), dtype=np.float32)[slice_index],
                artifacts.membership,
            )
            for slice_index, slice_id in enumerate(slice_ids)
        }
    candidate_mixtures = _iter_recommendation_candidates(str(input_bundle_root), round_count, baseline_mixture)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        candidate_expected_weight_sets = {
            candidate_id: _mixture_to_sample_weights(mixture, artifacts.membership)
            for candidate_id, mixture in candidate_mixtures.items()
        }
    candidate_manifest_dirs = [os.path.abspath(path) for path in args.candidate_manifest_dir]
    candidate_realized_weight_sets = _resolve_candidate_realized_weights(
        candidate_ids=candidate_mixtures.keys(),
        candidate_manifest_dirs=candidate_manifest_dirs,
        sample_ids=artifacts.sample_ids,
    )

    field_distributions = _build_field_distributions(
        schema=schema,
        feature_groups=feature_groups,
        feature_label_map=feature_label_map,
        sample_weight_sets=sample_weight_sets,
        slice_weight_sets=slice_weight_sets,
        candidate_expected_weight_sets=candidate_expected_weight_sets,
        candidate_realized_weight_sets=candidate_realized_weight_sets,
    )

    slice_report_dir = Path(args.slice_report_dir).resolve()
    sample_url_prefix = _infer_sample_url_prefix(slice_report_dir)
    slice_report_slices = _load_json(slice_report_dir / "slices.json")
    slice_report_samples = _load_json(slice_report_dir / "samples.json")
    slice_centers_2d = _load_json(slice_report_dir / "slice_centers_2d.json")
    if not isinstance(slice_report_slices, list) or not isinstance(slice_report_samples, list) or not isinstance(slice_centers_2d, list):
        raise ValueError("slice report bundle files must be JSON arrays")

    proxy_by_slice, proxy_pool, proxy_baseline = _compute_summary_proxy_values(
        schema=schema,
        feature_groups=feature_groups,
        feature_label_map=feature_label_map,
        slice_portraits=slice_portraits,
        memberships=artifacts.membership,
        baseline_weights=sample_weight_sets["baseline"],
    )

    slice_atlas = _build_slice_atlas(
        slice_report_slices=slice_report_slices,
        prior_graph_payload=prior_graph,
        proxy_by_slice=proxy_by_slice,
    )
    baseline_result_payload = _load_json(args.baseline_result_path) if args.baseline_result_path else None
    baseline_footprint = _build_baseline_footprint(
        baseline_seed=baseline_seed,
        budget=budget,
        baseline_mixture=baseline_mixture,
        pool_mixture=pool_mixture,
        proxy_by_slice=proxy_by_slice,
        proxy_pool=proxy_pool,
        proxy_baseline=proxy_baseline,
        baseline_result_payload=baseline_result_payload if isinstance(baseline_result_payload, dict) else None,
        metric_path=args.metric_path,
    )
    samples_payload = _build_samples_payload(
        slice_report_samples=slice_report_samples,
        slice_atlas_payload=slice_atlas,
        sample_url_prefix=sample_url_prefix,
    )
    slice_relationships = _build_slice_relationships(
        slice_ids=slice_ids,
        prior_graph_payload=prior_graph,
        slice_report_centers_2d=slice_centers_2d,
        slice_portraits=slice_portraits,
    )

    for output_root in args.output_root:
        root = Path(output_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / "prior_graph.json", prior_graph)
        _write_json(root / "slice_atlas.json", slice_atlas)
        _write_json(root / "baseline_footprint.json", baseline_footprint)
        _write_json(root / "field_distributions.json", field_distributions)
        _write_json(root / "samples.json", samples_payload)
        _write_json(root / "slice_relationships.json", slice_relationships)
        log_fn(f"wrote real workbench bundle to {root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
