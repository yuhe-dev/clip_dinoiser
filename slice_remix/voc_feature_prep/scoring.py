from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
from PIL import Image

from clip_dinoiser.feature_utils.data_feature.implementations.difficulty import (
    SmallObjectRatioCOCOStuff,
    cv2,
)
from clip_dinoiser.slice_remix.class_coverage import load_class_presence_matrix

from .contracts import (
    VOC_FOREGROUND_CLASSES,
    VocFeatureAxisDefinition,
    VocFeatureComputationResult,
    VocTrainAugRecord,
)
from .dataset import load_mask_array


DEFAULT_AXIS_DEFINITIONS: dict[str, VocFeatureAxisDefinition] = {
    "small_object_ratio": VocFeatureAxisDefinition(
        key="small_object_ratio",
        description="Fraction of connected components below a per-image area threshold.",
        family="difficulty",
        enabled_by_default=True,
    ),
    "rare_class_coverage": VocFeatureAxisDefinition(
        key="rare_class_coverage",
        description="Rarity-weighted foreground class exposure per image.",
        family="coverage",
        enabled_by_default=True,
    ),
    "rare_class_exposure_clipped": VocFeatureAxisDefinition(
        key="rare_class_exposure_clipped",
        description=(
            "Foreground class exposure using log inverse class-frequency weights, "
            "percentile clipping, and positive-mean normalization."
        ),
        family="coverage",
    ),
    "crop_survival_score": VocFeatureAxisDefinition(
        key="crop_survival_score",
        description=(
            "Monte Carlo estimate of foreground-mask survival under the supervised "
            "probe resize ratio range and 512 crop."
        ),
        family="difficulty",
    ),
    "foreground_class_count": VocFeatureAxisDefinition(
        key="foreground_class_count",
        description="Number of foreground classes present in the image.",
        family="coverage",
    ),
    "pixel_class_entropy": VocFeatureAxisDefinition(
        key="pixel_class_entropy",
        description="Entropy of the foreground pixel class distribution.",
        family="coverage",
    ),
    "foreground_area_ratio": VocFeatureAxisDefinition(
        key="foreground_area_ratio",
        description="Fraction of valid pixels assigned to foreground classes.",
        family="coverage",
    ),
    "foreground_component_count": VocFeatureAxisDefinition(
        key="foreground_component_count",
        description="Total number of connected foreground components across all classes.",
        family="difficulty",
    ),
    "component_fragmentation": VocFeatureAxisDefinition(
        key="component_fragmentation",
        description="Average number of connected components per present foreground class.",
        family="difficulty",
    ),
}


def available_feature_axes() -> tuple[str, ...]:
    return tuple(DEFAULT_AXIS_DEFINITIONS.keys())


def default_feature_axes() -> tuple[str, ...]:
    return tuple(
        axis_name
        for axis_name, axis_def in DEFAULT_AXIS_DEFINITIONS.items()
        if bool(axis_def.enabled_by_default)
    )


def resolve_feature_axes(feature_axes: Iterable[str] | None = None) -> tuple[str, ...]:
    if feature_axes is None:
        return default_feature_axes()

    ordered: list[str] = []
    for axis_name in feature_axes:
        key = str(axis_name)
        if key not in DEFAULT_AXIS_DEFINITIONS:
            supported = ", ".join(sorted(DEFAULT_AXIS_DEFINITIONS))
            raise ValueError(f"Unsupported VOC feature axis: {key} (supported: {supported})")
        if key not in ordered:
            ordered.append(key)
    if not ordered:
        raise ValueError("At least one feature axis must be requested.")
    return tuple(ordered)


def _normalize_positive_mean(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    positive = array[array > 0]
    if positive.size == 0:
        return np.ones_like(array, dtype=np.float32)
    scale = float(np.mean(positive))
    if scale <= 1e-8:
        return np.ones_like(array, dtype=np.float32)
    return (array / scale).astype(np.float32)


def _clipped_log_inverse_class_weights(
    class_presence_rate: np.ndarray,
    *,
    clip_percentile: float,
) -> np.ndarray:
    clipped_percentile = float(np.clip(float(clip_percentile), 0.0, 100.0))
    safe_rate = np.clip(np.asarray(class_presence_rate, dtype=np.float32), 1e-6, 1.0)
    log_inverse = np.log(1.0 / safe_rate).astype(np.float32)
    positive = log_inverse[log_inverse > 0]
    if positive.size > 0:
        ceiling = float(np.percentile(positive, clipped_percentile))
        log_inverse = np.minimum(log_inverse, ceiling).astype(np.float32)
    return _normalize_positive_mean(log_inverse)


def _compute_class_presence_bundle(
    records: list[VocTrainAugRecord],
    *,
    data_root: str,
    rare_class_clip_percentile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = load_class_presence_matrix(
        sample_ids=[record.image_rel for record in records],
        annotation_root=str(data_root),
        annotation_rels=[record.annotation_rel for record in records],
        annotation_suffix=".png",
        num_classes=len(VOC_FOREGROUND_CLASSES),
        reduce_zero_label=True,
    ).astype(np.uint8)
    class_presence_rate = matrix.mean(axis=0).astype(np.float32)
    raw_weights = 1.0 / np.clip(class_presence_rate, 1e-6, None)
    rarity_weights = _normalize_positive_mean(raw_weights)
    clipped_rarity_weights = _clipped_log_inverse_class_weights(
        class_presence_rate,
        clip_percentile=float(rare_class_clip_percentile),
    )
    return matrix, class_presence_rate, rarity_weights, clipped_rarity_weights


def _stable_uint32(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _compute_small_object_scores(
    records: list[VocTrainAugRecord],
    *,
    data_root: str,
    tau_ratio: float,
) -> np.ndarray:
    scorer = SmallObjectRatioCOCOStuff(
        tau_ratio=float(tau_ratio),
        thing_id_start=1,
        num_things=20,
        default_ignore_index=255,
        use_things_only=True,
    )
    scores = np.zeros(len(records), dtype=np.float32)
    for index, record in enumerate(records):
        mask = load_mask_array(record.annotation_path(data_root))
        scores[index] = float(
            scorer.get_score(
                np.empty((1, 1, 3), dtype=np.uint8),
                mask=mask,
                meta={"thing_ids": list(range(1, 21)), "ignore_index": 255},
            )
        )
    return scores


def _resize_mask_keep_ratio(mask: np.ndarray, *, target_size: int, ratio: float) -> np.ndarray:
    height, width = mask.shape[:2]
    if height <= 0 or width <= 0:
        return np.asarray(mask, dtype=np.uint8)
    scaled_target = max(1.0, float(target_size) * float(ratio))
    scale = min(scaled_target / float(height), scaled_target / float(width))
    resized_height = max(1, int(round(float(height) * scale)))
    resized_width = max(1, int(round(float(width) * scale)))
    if resized_height == height and resized_width == width:
        return np.asarray(mask, dtype=np.uint8)
    if hasattr(cv2, "resize"):
        return cv2.resize(
            np.asarray(mask, dtype=np.uint8),
            (int(resized_width), int(resized_height)),
            interpolation=cv2.INTER_NEAREST,
        )
    pil_mask = Image.fromarray(np.asarray(mask, dtype=np.uint8))
    resized = pil_mask.resize((int(resized_width), int(resized_height)), resample=Image.NEAREST)
    return np.asarray(resized, dtype=np.uint8)


def _random_crop_mask(
    mask: np.ndarray,
    *,
    crop_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = mask.shape[:2]
    top_margin = max(int(height) - int(crop_size), 0)
    left_margin = max(int(width) - int(crop_size), 0)
    top = int(rng.integers(0, top_margin + 1)) if top_margin > 0 else 0
    left = int(rng.integers(0, left_margin + 1)) if left_margin > 0 else 0
    return mask[top : min(top + int(crop_size), height), left : min(left + int(crop_size), width)]


def _foreground_pixel_count(mask: np.ndarray) -> int:
    array = np.asarray(mask, dtype=np.int32)
    return int(((array > 0) & (array != 255)).sum())


def _compute_crop_survival_scores(
    records: list[VocTrainAugRecord],
    *,
    data_root: str,
    crop_size: int,
    resize_ratio_min: float,
    resize_ratio_max: float,
    simulations: int,
    seed: int,
) -> np.ndarray:
    resolved_simulations = max(1, int(simulations))
    resolved_crop_size = max(1, int(crop_size))
    ratio_min = float(min(resize_ratio_min, resize_ratio_max))
    ratio_max = float(max(resize_ratio_min, resize_ratio_max))
    scores = np.zeros(len(records), dtype=np.float32)

    for index, record in enumerate(records):
        mask = load_mask_array(record.annotation_path(data_root))
        original_foreground = _foreground_pixel_count(mask)
        if original_foreground <= 0:
            scores[index] = 0.0
            continue

        rng = np.random.default_rng(int(seed) + _stable_uint32(record.stem))
        trial_scores = np.zeros(resolved_simulations, dtype=np.float32)
        for trial_index in range(resolved_simulations):
            ratio = float(rng.uniform(ratio_min, ratio_max))
            resized = _resize_mask_keep_ratio(mask, target_size=resolved_crop_size, ratio=ratio)
            resized_foreground = _foreground_pixel_count(resized)
            if resized_foreground <= 0:
                trial_scores[trial_index] = 0.0
                continue
            cropped = _random_crop_mask(resized, crop_size=resolved_crop_size, rng=rng)
            trial_scores[trial_index] = float(_foreground_pixel_count(cropped) / float(resized_foreground))
        scores[index] = float(np.mean(trial_scores))
    return scores.astype(np.float32)


def _foreground_labels(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask, dtype=np.int32)
    labels = np.unique(array)
    labels = labels[(labels > 0) & (labels != 255)]
    return labels.astype(np.int32)


def _compute_foreground_area_ratio(mask: np.ndarray) -> float:
    array = np.asarray(mask, dtype=np.int32)
    valid = array != 255
    valid_count = int(valid.sum())
    if valid_count == 0:
        return 0.0
    foreground = (array > 0) & valid
    return float(foreground.sum() / float(valid_count))


def _compute_pixel_class_entropy(mask: np.ndarray) -> float:
    array = np.asarray(mask, dtype=np.int32)
    foreground = array[(array > 0) & (array != 255)]
    if foreground.size == 0:
        return 0.0
    labels, counts = np.unique(foreground, return_counts=True)
    if labels.size <= 1:
        return 0.0
    probs = counts.astype(np.float32) / float(counts.sum())
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), dtype=np.float32)
    return float(entropy)


def _count_foreground_components(mask: np.ndarray) -> tuple[int, float]:
    array = np.asarray(mask, dtype=np.int32)
    labels = _foreground_labels(array)
    if labels.size == 0:
        return 0, 0.0

    component_count = 0
    for label in labels.tolist():
        binary = (array == int(label)).astype(np.uint8)
        num, _label_map, _stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num > 1:
            component_count += int(num - 1)
    fragmentation = float(component_count / float(labels.size)) if labels.size > 0 else 0.0
    return int(component_count), fragmentation


def compute_voc_feature_rows(
    records: list[VocTrainAugRecord],
    *,
    data_root: str,
    feature_axes: Iterable[str] | None = None,
    small_object_tau_ratio: float = 0.02,
    rare_class_clip_percentile: float = 95.0,
    crop_survival_crop_size: int = 512,
    crop_survival_resize_ratio_range: tuple[float, float] = (0.5, 2.0),
    crop_survival_simulations: int = 24,
    crop_survival_seed: int = 0,
) -> VocFeatureComputationResult:
    resolved_axes = resolve_feature_axes(feature_axes)
    (
        class_presence_matrix,
        class_presence_rate,
        rarity_weights,
        clipped_rarity_weights,
    ) = _compute_class_presence_bundle(
        records,
        data_root=data_root,
        rare_class_clip_percentile=float(rare_class_clip_percentile),
    )
    foreground_class_count = class_presence_matrix.sum(axis=1).astype(np.int64)

    axis_scores: dict[str, np.ndarray] = {}
    if "small_object_ratio" in resolved_axes:
        axis_scores["small_object_ratio"] = _compute_small_object_scores(
            records,
            data_root=data_root,
            tau_ratio=float(small_object_tau_ratio),
        ).astype(np.float32)

    if "rare_class_coverage" in resolved_axes:
        rare_class_scores = (
            class_presence_matrix.astype(np.float32) * rarity_weights[None, :]
        ).sum(axis=1).astype(np.float32)
        axis_scores["rare_class_coverage"] = rare_class_scores

    if "rare_class_exposure_clipped" in resolved_axes:
        clipped_rare_class_scores = (
            class_presence_matrix.astype(np.float32) * clipped_rarity_weights[None, :]
        ).sum(axis=1).astype(np.float32)
        axis_scores["rare_class_exposure_clipped"] = clipped_rare_class_scores

    if "crop_survival_score" in resolved_axes:
        axis_scores["crop_survival_score"] = _compute_crop_survival_scores(
            records,
            data_root=data_root,
            crop_size=int(crop_survival_crop_size),
            resize_ratio_min=float(crop_survival_resize_ratio_range[0]),
            resize_ratio_max=float(crop_survival_resize_ratio_range[1]),
            simulations=int(crop_survival_simulations),
            seed=int(crop_survival_seed),
        )

    if "foreground_class_count" in resolved_axes:
        axis_scores["foreground_class_count"] = foreground_class_count.astype(np.float32)

    need_mask_pass = any(
        axis_name in resolved_axes
        for axis_name in ("pixel_class_entropy", "foreground_area_ratio", "foreground_component_count", "component_fragmentation")
    )
    pixel_class_entropy = np.zeros(len(records), dtype=np.float32)
    foreground_area_ratio = np.zeros(len(records), dtype=np.float32)
    foreground_component_count = np.zeros(len(records), dtype=np.float32)
    component_fragmentation = np.zeros(len(records), dtype=np.float32)
    if need_mask_pass:
        for index, record in enumerate(records):
            mask = load_mask_array(record.annotation_path(data_root))
            if "pixel_class_entropy" in resolved_axes:
                pixel_class_entropy[index] = float(_compute_pixel_class_entropy(mask))
            if "foreground_area_ratio" in resolved_axes:
                foreground_area_ratio[index] = float(_compute_foreground_area_ratio(mask))
            if "foreground_component_count" in resolved_axes or "component_fragmentation" in resolved_axes:
                comp_count, fragmentation = _count_foreground_components(mask)
                foreground_component_count[index] = float(comp_count)
                component_fragmentation[index] = float(fragmentation)

    if "pixel_class_entropy" in resolved_axes:
        axis_scores["pixel_class_entropy"] = pixel_class_entropy

    if "foreground_area_ratio" in resolved_axes:
        axis_scores["foreground_area_ratio"] = foreground_area_ratio

    if "foreground_component_count" in resolved_axes:
        axis_scores["foreground_component_count"] = foreground_component_count

    if "component_fragmentation" in resolved_axes:
        axis_scores["component_fragmentation"] = component_fragmentation

    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        present_mask = class_presence_matrix[index].astype(bool)
        present_classes = [
            VOC_FOREGROUND_CLASSES[class_index]
            for class_index, flag in enumerate(present_mask.tolist())
            if flag
        ]
        row: dict[str, object] = {
            "stem": record.stem,
            "image_rel": record.image_rel,
            "annotation_rel": record.annotation_rel,
            "image_path": record.image_path(data_root),
            "annotation_path": record.annotation_path(data_root),
            "foreground_class_count": int(foreground_class_count[index]),
            "present_classes": present_classes,
        }
        for axis_name, values in axis_scores.items():
            row[axis_name] = float(values[index])
        rows.append(row)

    return VocFeatureComputationResult(
        axis_scores=axis_scores,
        rows=rows,
        class_presence_matrix=class_presence_matrix,
        class_presence_rate=class_presence_rate,
        rarity_weights=rarity_weights,
        clipped_rarity_weights=clipped_rarity_weights,
        foreground_class_count=foreground_class_count,
    )


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return {"min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25.0)),
        "p50": float(np.percentile(array, 50.0)),
        "p75": float(np.percentile(array, 75.0)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }
