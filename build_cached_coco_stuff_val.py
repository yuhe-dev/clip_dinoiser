from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from slice_remix.eval_cache import build_cache_record, resolve_keep_ratio_size

try:
    import cv2
except ImportError:  # pragma: no cover - optional local fallback
    cv2 = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a cached full-slide COCO-Stuff validation set.")
    parser.add_argument("--data-root", default="./data/coco_stuff164k")
    parser.add_argument("--output-root", default="./data/coco_stuff164k_eval_cache_slide")
    parser.add_argument("--max-long-edge", type=int, default=2048)
    parser.add_argument("--max-short-edge", type=int, default=448)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def resolve_mask_name(image_name: str) -> str:
    stem = Path(image_name).stem
    return f"{stem}_labelTrainIds.png"


def _resize_rgb_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    pil_image = Image.fromarray(image)
    return np.asarray(pil_image.resize((width, height), resample=Image.BILINEAR))


def _write_meta(output_root: str, *, data_root: str, img_scale: tuple[int, int], count: int) -> None:
    payload = {
        "source_data_root": os.path.abspath(data_root),
        "img_scale": [int(v) for v in img_scale],
        "keep_ratio": True,
        "test_cfg": {"mode": "slide", "stride": [224, 224], "crop_size": [448, 448]},
        "count": int(count),
    }
    with open(os.path.join(output_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_cache(
    *,
    data_root: str,
    output_root: str,
    img_scale: tuple[int, int],
    limit: int = 0,
) -> int:
    image_dir = os.path.join(data_root, "images", "val2017")
    mask_dir = os.path.join(data_root, "annotations", "val2017")
    output_images = os.path.join(output_root, "images")
    output_masks = os.path.join(output_root, "masks")
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_masks, exist_ok=True)

    image_names = sorted(name for name in os.listdir(image_dir) if name.lower().endswith(".jpg"))
    if limit > 0:
        image_names = image_names[: int(limit)]

    manifest_path = os.path.join(output_root, "manifest.jsonl")
    written = 0
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        for image_name in image_names:
            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, resolve_mask_name(image_name))
            if not os.path.exists(mask_path):
                raise FileNotFoundError(mask_path)

            image = np.asarray(Image.open(image_path).convert("RGB"))
            mask = np.asarray(Image.open(mask_path))
            resized_height, resized_width = resolve_keep_ratio_size(
                height=int(image.shape[0]),
                width=int(image.shape[1]),
                scale=img_scale,
            )
            resized = _resize_rgb_image(image, resized_width, resized_height)

            stem = Path(image_name).stem
            image_rel_path = os.path.join("images", f"{stem}.npy")
            mask_rel_path = os.path.join("masks", f"{stem}.npy")
            np.save(os.path.join(output_root, image_rel_path), resized.astype(np.uint8, copy=False))
            np.save(os.path.join(output_root, mask_rel_path), mask.astype(np.uint8, copy=False))

            width_scale = float(resized_width) / float(image.shape[1])
            height_scale = float(resized_height) / float(image.shape[0])
            record = build_cache_record(
                basename=image_name,
                image_rel_path=image_rel_path,
                mask_rel_path=mask_rel_path,
                ori_shape=(int(image.shape[0]), int(image.shape[1]), int(image.shape[2])),
                cached_img_shape=(int(resized.shape[0]), int(resized.shape[1]), int(resized.shape[2])),
                scale_factor=(width_scale, height_scale, width_scale, height_scale),
            )
            manifest_file.write(json.dumps(record) + "\n")
            written += 1

    _write_meta(output_root, data_root=data_root, img_scale=img_scale, count=written)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    img_scale = (int(args.max_long_edge), int(args.max_short_edge))
    written = build_cache(
        data_root=os.path.abspath(args.data_root),
        output_root=os.path.abspath(args.output_root),
        img_scale=img_scale,
        limit=int(args.limit),
    )
    print(
        f"built cached COCO-Stuff val cache: output_root={os.path.abspath(args.output_root)} count={written}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
