from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image


def resolve_annotation_path(sample_id: str, annotation_root: str) -> str:
    normalized_sample = str(sample_id).replace("\\", "/")
    stem, _ext = os.path.splitext(os.path.basename(normalized_sample))
    filename = f"{stem}_labelTrainIds.png"

    normalized_root = os.path.abspath(annotation_root)
    rel_parts = normalized_sample.split("/")
    if len(rel_parts) >= 3 and rel_parts[0] == "images":
        split_name = rel_parts[1]
        if os.path.basename(normalized_root) == split_name:
            return os.path.join(normalized_root, filename)
        return os.path.join(normalized_root, "annotations", split_name, filename)
    return os.path.join(normalized_root, filename)


def load_class_presence_matrix(
    sample_ids: list[str],
    annotation_root: str,
    *,
    num_classes: int,
    ignore_label: int = 255,
) -> np.ndarray:
    matrix = np.zeros((len(sample_ids), int(num_classes)), dtype=np.uint8)
    for index, sample_id in enumerate(sample_ids):
        annotation_path = resolve_annotation_path(sample_id, annotation_root)
        mask = np.asarray(Image.open(annotation_path), dtype=np.int64)
        labels = np.unique(mask)
        for label in labels.tolist():
            if int(label) == int(ignore_label):
                continue
            if 0 <= int(label) < int(num_classes):
                matrix[index, int(label)] = 1
    return matrix


def select_focus_class_spec(
    *,
    baseline_result: dict[str, Any],
    full_result: dict[str, Any],
    task_key: str = "coco_stuff",
    min_iou_gap: float = 10.0,
    top_k: int = 25,
) -> dict[str, object]:
    baseline_per_class = dict((baseline_result.get(task_key) or {}).get("per_class") or {})
    full_per_class = dict((full_result.get(task_key) or {}).get("per_class") or {})
    if not baseline_per_class or not full_per_class:
        return {"class_indices": [], "class_names": [], "class_weights": []}

    rows: list[tuple[float, int, str]] = []
    ordered_names = list(full_per_class.keys())
    for index, class_name in enumerate(ordered_names):
        full_iou = float((full_per_class.get(class_name) or {}).get("IoU", 0.0))
        base_iou = float((baseline_per_class.get(class_name) or {}).get("IoU", 0.0))
        gap = full_iou - base_iou
        if gap >= float(min_iou_gap):
            rows.append((gap, index, class_name))

    rows.sort(reverse=True)
    rows = rows[: int(top_k)]
    if not rows:
        return {"class_indices": [], "class_names": [], "class_weights": []}

    gaps = np.asarray([row[0] for row in rows], dtype=np.float32)
    weights = gaps / float(gaps.sum()) if float(gaps.sum()) > 0.0 else np.ones_like(gaps) / float(len(gaps))
    return {
        "class_indices": [int(row[1]) for row in rows],
        "class_names": [str(row[2]) for row in rows],
        "class_weights": [float(value) for value in weights.tolist()],
    }
