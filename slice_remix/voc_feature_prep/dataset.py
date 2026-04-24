from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from clip_dinoiser.tools.sample_voc20_subset import build_split_records

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
    records = build_split_records(
        root,
        split_path,
        annotation_dir="SegmentationClassAug",
        annotation_suffix=".png",
    )
    return [
        VocTrainAugRecord(
            stem=str(record["stem"]),
            image_rel=str(record["image_rel"]),
            annotation_rel=str(record["annotation_rel"]),
        )
        for record in records
    ]


def load_mask_array(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.uint8)
