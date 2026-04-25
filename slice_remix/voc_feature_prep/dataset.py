from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from .contracts import VocTrainAugRecord


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent


def _existing_root(*candidates: str) -> str:
    resolved = [os.path.abspath(str(candidate)) for candidate in candidates]
    for candidate in resolved:
        if os.path.exists(candidate):
            return candidate
    return resolved[0]


DEFAULT_VOC_ROOT = _existing_root(
    os.path.join(str(REPO_ROOT), "data", "VOCdevkit", "VOC2012"),
    os.path.join(
        str(WORKSPACE_ROOT),
        "deeplab",
        "research",
        "deeplab",
        "datasets",
        "pascal_voc_seg",
        "VOCdevkit",
        "VOC2012",
    ),
)


def build_voc_train_aug_records(data_root: str | Path) -> list[VocTrainAugRecord]:
    root = Path(data_root).resolve()
    split_path = root / "ImageSets" / "Segmentation" / "train_aug.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    records: list[VocTrainAugRecord] = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        stem = line.strip()
        if not stem:
            continue
        image_path = root / "JPEGImages" / f"{stem}.jpg"
        annotation_path = root / "SegmentationClassAug" / f"{stem}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found for split entry '{stem}': {image_path}")
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Annotation not found for split entry '{stem}': {annotation_path}")
        records.append(
            VocTrainAugRecord(
                stem=stem,
                image_rel=image_path.relative_to(root).as_posix(),
                annotation_rel=annotation_path.relative_to(root).as_posix(),
            )
        )
    return records


def load_mask_array(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.uint8)
