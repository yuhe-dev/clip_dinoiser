from __future__ import annotations

import json
from typing import Any


def resolve_keep_ratio_size(*, height: int, width: int, scale: tuple[int, int]) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    max_long_edge, max_short_edge = int(scale[0]), int(scale[1])
    long_edge = max(height, width)
    short_edge = min(height, width)
    resize_ratio = min(max_long_edge / float(long_edge), max_short_edge / float(short_edge))
    resized_height = max(1, int(round(height * resize_ratio)))
    resized_width = max(1, int(round(width * resize_ratio)))
    return resized_height, resized_width


def build_cache_record(
    *,
    basename: str,
    image_rel_path: str,
    mask_rel_path: str,
    ori_shape: tuple[int, int, int],
    cached_img_shape: tuple[int, int, int],
    scale_factor: tuple[float, float, float, float],
) -> dict[str, Any]:
    return {
        "basename": str(basename),
        "image_npy": str(image_rel_path),
        "mask_npy": str(mask_rel_path),
        "ori_shape": [int(value) for value in ori_shape],
        "cached_img_shape": [int(value) for value in cached_img_shape],
        "scale_factor": [float(value) for value in scale_factor],
    }


def load_cache_manifest(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(dict(json.loads(stripped)))
    return rows
