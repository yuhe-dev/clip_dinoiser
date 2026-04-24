from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .contracts import (
    VOC_FOREGROUND_CLASSES,
    VocFeaturePreparationArtifacts,
    VocTrainAugRecord,
)
from .dataset import DEFAULT_VOC_ROOT, build_voc_train_aug_records
from .scoring import compute_voc_feature_rows, percentile_summary, resolve_feature_axes


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return target


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


def _build_manifest(
    *,
    output_dir: str | Path,
    candidate_id: str,
    records: list[VocTrainAugRecord],
    indices: np.ndarray,
    metadata: dict[str, Any],
    data_root: str | Path,
) -> Path:
    manifest_path = Path(output_dir) / "manifests" / f"{candidate_id}.json"
    selected = [records[int(index)] for index in indices.tolist()]
    payload = {
        "candidate_id": candidate_id,
        "sample_ids": [record.stem for record in selected],
        "sample_paths": [record.image_path(data_root) for record in selected],
        "metadata": metadata,
    }
    return _write_json(manifest_path, payload)


def _anchor_axis_means(axis_scores: dict[str, np.ndarray], anchor_indices: np.ndarray) -> dict[str, float]:
    return {
        axis_name: float(values[anchor_indices].mean())
        for axis_name, values in axis_scores.items()
    }


def _select_extreme_subset(
    *,
    axis_name: str,
    direction: int,
    axis_scores: dict[str, np.ndarray],
    anchor_indices: np.ndarray,
    class_presence_matrix: np.ndarray,
    candidate_budget: int,
    subset_size: int,
    rng: np.random.Generator,
    excluded: set[int] | None = None,
) -> np.ndarray:
    excluded = excluded or set()
    target_scores = axis_scores[axis_name]
    anchor_means = _anchor_axis_means(axis_scores, anchor_indices)
    anchor_class_mean = class_presence_matrix[anchor_indices].mean(axis=0).astype(np.float32)
    non_target_axes = [name for name in axis_scores if name != axis_name]

    base_order = np.argsort(-(direction * target_scores))
    candidate_indices = [
        int(index)
        for index in base_order.tolist()
        if int(index) not in excluded
    ]
    candidate_indices = np.asarray(candidate_indices[:candidate_budget], dtype=np.int64)
    if candidate_indices.size < subset_size:
        raise ValueError(
            f"candidate pool underfilled for axis={axis_name} direction={direction}: "
            f"{candidate_indices.size} < {subset_size}"
        )

    penalties = np.zeros(candidate_indices.shape[0], dtype=np.float32)
    if non_target_axes:
        penalties += 0.75 * np.mean(
            np.stack(
                [
                    np.abs(axis_scores[name][candidate_indices] - anchor_means[name]).astype(np.float32)
                    for name in non_target_axes
                ],
                axis=0,
            ),
            axis=0,
        )
    class_penalty = np.mean(
        np.abs(class_presence_matrix[candidate_indices].astype(np.float32) - anchor_class_mean[None, :]),
        axis=1,
    )
    penalties += 3.0 * class_penalty.astype(np.float32)
    jitter = rng.normal(0.0, 1e-6, size=candidate_indices.shape[0]).astype(np.float32)
    composite = (direction * target_scores[candidate_indices]).astype(np.float32) - penalties + jitter
    ordered = candidate_indices[np.argsort(-composite)]
    return np.sort(ordered[:subset_size].astype(np.int64))


def _select_matched_random_subset(
    *,
    axis_name: str,
    axis_scores: dict[str, np.ndarray],
    anchor_indices: np.ndarray,
    class_presence_matrix: np.ndarray,
    candidate_budget: int,
    subset_size: int,
    seed: int,
    excluded: set[int] | None = None,
) -> np.ndarray:
    excluded = excluded or set()
    anchor_means = _anchor_axis_means(axis_scores, anchor_indices)
    anchor_class_mean = class_presence_matrix[anchor_indices].mean(axis=0).astype(np.float32)
    non_target_axes = [name for name in axis_scores if name != axis_name]

    all_indices = [index for index in range(class_presence_matrix.shape[0]) if int(index) not in excluded]
    candidate_indices = np.asarray(all_indices, dtype=np.int64)
    penalties = np.zeros(candidate_indices.shape[0], dtype=np.float32)
    if non_target_axes:
        penalties += np.mean(
            np.stack(
                [
                    np.abs(axis_scores[name][candidate_indices] - anchor_means[name]).astype(np.float32)
                    for name in non_target_axes
                ],
                axis=0,
            ),
            axis=0,
        )
    class_penalty = np.mean(
        np.abs(class_presence_matrix[candidate_indices].astype(np.float32) - anchor_class_mean[None, :]),
        axis=1,
    )
    penalties += 3.0 * class_penalty.astype(np.float32)
    ordered = candidate_indices[np.argsort(penalties)]
    pool = ordered[: max(subset_size, candidate_budget)]
    if pool.shape[0] < subset_size:
        raise ValueError(f"matched random pool underfilled for axis={axis_name}: {pool.shape[0]} < {subset_size}")
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(pool, size=subset_size, replace=False)
    return np.sort(selected.astype(np.int64))


def _overlap_count(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.intersect1d(left.astype(np.int64), right.astype(np.int64)).shape[0])


def prepare_voc_train_aug_feature_experiment(
    *,
    data_root: str | None = None,
    output_dir: str,
    subset_size: int = 2000,
    anchor_seed: int = 0,
    candidate_budget: int | None = None,
    small_object_tau_ratio: float = 0.02,
    feature_axes: Iterable[str] | None = None,
) -> dict[str, Any]:
    resolved_root = os.path.abspath(data_root or DEFAULT_VOC_ROOT)
    resolved_axes = resolve_feature_axes(feature_axes)
    records = build_voc_train_aug_records(resolved_root)

    if subset_size <= 0:
        raise ValueError("subset_size must be positive")
    if subset_size >= len(records):
        raise ValueError(f"subset_size must be smaller than pool size ({len(records)})")

    pool_size = len(records)
    max_candidate_budget = max(pool_size - subset_size, 0)
    resolved_candidate_budget = int(candidate_budget) if candidate_budget is not None else max(subset_size * 5, 5000)
    resolved_candidate_budget = max(subset_size, min(max_candidate_budget, resolved_candidate_budget))
    if resolved_candidate_budget < subset_size:
        raise ValueError("candidate_budget must be at least subset_size after clipping")

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    computation = compute_voc_feature_rows(
        records,
        data_root=resolved_root,
        feature_axes=resolved_axes,
        small_object_tau_ratio=float(small_object_tau_ratio),
    )
    feature_table_path = _write_jsonl(output_root / "feature_table.jsonl", computation.rows)

    rng = np.random.default_rng(int(anchor_seed))
    anchor_indices = np.sort(rng.choice(pool_size, size=subset_size, replace=False).astype(np.int64))
    anchor_candidate_id = f"voc_train_aug_anchor_{subset_size}_seed{anchor_seed}"
    anchor_manifest_path = _build_manifest(
        output_dir=output_root,
        candidate_id=anchor_candidate_id,
        records=records,
        indices=anchor_indices,
        metadata={
            "kind": "anchor",
            "subset_size": int(subset_size),
            "anchor_seed": int(anchor_seed),
            "pool_size": int(pool_size),
            "feature_axes": list(resolved_axes),
        },
        data_root=resolved_root,
    )

    manifest_index: dict[str, str] = {"anchor": str(anchor_manifest_path)}
    axis_summary: dict[str, Any] = {}
    anchor_excluded = set(int(index) for index in anchor_indices.tolist())
    anchor_feature_means = _anchor_axis_means(computation.axis_scores, anchor_indices)

    for axis_offset, axis_name in enumerate(resolved_axes):
        local_seed = int(anchor_seed + axis_offset + 1)
        local_rng = np.random.default_rng(local_seed)
        high_indices = _select_extreme_subset(
            axis_name=axis_name,
            direction=1,
            axis_scores=computation.axis_scores,
            anchor_indices=anchor_indices,
            class_presence_matrix=computation.class_presence_matrix,
            candidate_budget=resolved_candidate_budget,
            subset_size=subset_size,
            rng=local_rng,
            excluded=set(anchor_excluded),
        )
        low_indices = _select_extreme_subset(
            axis_name=axis_name,
            direction=-1,
            axis_scores=computation.axis_scores,
            anchor_indices=anchor_indices,
            class_presence_matrix=computation.class_presence_matrix,
            candidate_budget=resolved_candidate_budget,
            subset_size=subset_size,
            rng=local_rng,
            excluded=set(anchor_excluded | set(int(index) for index in high_indices.tolist())),
        )
        matched_random_indices = _select_matched_random_subset(
            axis_name=axis_name,
            axis_scores=computation.axis_scores,
            anchor_indices=anchor_indices,
            class_presence_matrix=computation.class_presence_matrix,
            candidate_budget=resolved_candidate_budget,
            subset_size=subset_size,
            seed=local_seed,
            excluded=set(
                anchor_excluded
                | set(int(index) for index in high_indices.tolist())
                | set(int(index) for index in low_indices.tolist())
            ),
        )

        high_candidate_id = f"voc_train_aug_{axis_name}_high_{subset_size}_seed{anchor_seed}"
        low_candidate_id = f"voc_train_aug_{axis_name}_low_{subset_size}_seed{anchor_seed}"
        matched_random_candidate_id = f"voc_train_aug_{axis_name}_matched_random_{subset_size}_seed{anchor_seed}"

        high_manifest_path = _build_manifest(
            output_dir=output_root,
            candidate_id=high_candidate_id,
            records=records,
            indices=high_indices,
            metadata={
                "kind": "high",
                "axis": axis_name,
                "subset_size": int(subset_size),
                "anchor_seed": int(anchor_seed),
                "candidate_budget": int(resolved_candidate_budget),
                "target_mean": float(computation.axis_scores[axis_name][high_indices].mean()),
            },
            data_root=resolved_root,
        )
        low_manifest_path = _build_manifest(
            output_dir=output_root,
            candidate_id=low_candidate_id,
            records=records,
            indices=low_indices,
            metadata={
                "kind": "low",
                "axis": axis_name,
                "subset_size": int(subset_size),
                "anchor_seed": int(anchor_seed),
                "candidate_budget": int(resolved_candidate_budget),
                "target_mean": float(computation.axis_scores[axis_name][low_indices].mean()),
            },
            data_root=resolved_root,
        )
        matched_random_manifest_path = _build_manifest(
            output_dir=output_root,
            candidate_id=matched_random_candidate_id,
            records=records,
            indices=matched_random_indices,
            metadata={
                "kind": "matched_random",
                "axis": axis_name,
                "subset_size": int(subset_size),
                "anchor_seed": int(anchor_seed),
                "candidate_budget": int(resolved_candidate_budget),
                "disjoint_with": ["anchor", "high", "low"],
                "target_mean": float(computation.axis_scores[axis_name][matched_random_indices].mean()),
            },
            data_root=resolved_root,
        )

        manifest_index[f"{axis_name}.high"] = str(high_manifest_path)
        manifest_index[f"{axis_name}.low"] = str(low_manifest_path)
        manifest_index[f"{axis_name}.matched_random"] = str(matched_random_manifest_path)
        axis_summary[axis_name] = {
            "global": percentile_summary(computation.axis_scores[axis_name]),
            "anchor_mean": float(anchor_feature_means[axis_name]),
            "high_mean": float(computation.axis_scores[axis_name][high_indices].mean()),
            "low_mean": float(computation.axis_scores[axis_name][low_indices].mean()),
            "matched_random_mean": float(computation.axis_scores[axis_name][matched_random_indices].mean()),
            "high_minus_low": float(
                computation.axis_scores[axis_name][high_indices].mean()
                - computation.axis_scores[axis_name][low_indices].mean()
            ),
            "matched_random_minus_anchor": float(
                computation.axis_scores[axis_name][matched_random_indices].mean()
                - computation.axis_scores[axis_name][anchor_indices].mean()
            ),
            "overlap_counts": {
                "anchor": _overlap_count(matched_random_indices, anchor_indices),
                "high": _overlap_count(matched_random_indices, high_indices),
                "low": _overlap_count(matched_random_indices, low_indices),
            },
        }

    manifest_index_path = _write_json(output_root / "manifest_index.json", manifest_index)
    summary_path = _write_json(
        output_root / "summary.json",
        {
            "data_root": resolved_root,
            "split": "ImageSets/Segmentation/train_aug.txt",
            "pool_size": int(pool_size),
            "subset_size": int(subset_size),
            "anchor_seed": int(anchor_seed),
            "candidate_budget": int(resolved_candidate_budget),
            "small_object_tau_ratio": float(small_object_tau_ratio),
            "feature_axes": list(resolved_axes),
            "axis_summary": axis_summary,
            "rare_class_stats": {
                "class_presence_rate": {
                    class_name: float(computation.class_presence_rate[index])
                    for index, class_name in enumerate(VOC_FOREGROUND_CLASSES)
                },
                "class_rarity_weight": {
                    class_name: float(computation.rarity_weights[index])
                    for index, class_name in enumerate(VOC_FOREGROUND_CLASSES)
                },
            },
        },
    )

    artifacts = VocFeaturePreparationArtifacts(
        data_root=resolved_root,
        subset_size=int(subset_size),
        feature_axes=resolved_axes,
        feature_table_path=str(feature_table_path),
        summary_path=str(summary_path),
        manifest_index_path=str(manifest_index_path),
        manifest_paths=manifest_index,
    )
    return artifacts.to_payload()
