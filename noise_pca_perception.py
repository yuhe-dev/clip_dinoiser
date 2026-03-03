import os
import argparse
import shutil
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


# TODO: 改成你工程里真实的 import 路径
# from feature_utils.data_feature.implementations.quality import WeakTexturePCANoise

# 如果你不想改 import，也可以把 WeakTexturePCANoise 类直接粘贴到这个脚本里
from feature_utils.data_feature.implementations.quality import WeakTexturePCANoise  # <-- 改这里


def list_images(img_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".webp")
    return [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(exts)
    ]


def read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def save_grid(image_paths: List[str], out_path: str, cols: int = 10, thumb: int = 192) -> None:
    if len(image_paths) == 0:
        return
    rows = int(np.ceil(len(image_paths) / cols))
    canvas = Image.new("RGB", (cols * thumb, rows * thumb), (255, 255, 255))

    for idx, p in enumerate(image_paths):
        r = idx // cols
        c = idx % cols
        try:
            im = Image.open(p).convert("RGB")
            im = im.resize((thumb, thumb))
            canvas.paste(im, (c * thumb, r * thumb))
        except Exception:
            # 读失败就空白占位
            pass

    canvas.save(out_path)


def export_ranked_images(
    scored: List[Tuple[str, float]],
    out_dir: str,
    k: int,
    prefix: str,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    exported_paths = []
    for rank, (path, score) in enumerate(scored[:k]):
        base = os.path.basename(path)
        # 文件名加上 rank 和 score，方便追溯
        new_name = f"{prefix}_{rank:04d}_sigma={score:.6f}_{base}"
        dst = os.path.join(out_dir, new_name)
        shutil.copy2(path, dst)
        exported_paths.append(dst)
    return exported_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="COCO-Stuff images dir, e.g., .../images/train2017")
    parser.add_argument("--out_dir", type=str, default="noise_pca_inspect", help="output directory")
    parser.add_argument("--max_n", type=int, default=2000, help="max images to score (for quick inspection)")
    parser.add_argument("--topk", type=int, default=50, help="top-k highest noise images to export")
    parser.add_argument("--bottomk", type=int, default=50, help="bottom-k lowest noise images to export")

    # noise_pca hyperparams (keep defaults unless you know why)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--weak_texture_percentile", type=float, default=10.0)
    parser.add_argument("--max_patches", type=int, default=5000)
    parser.add_argument("--tail_eig_k", type=int, default=1)
    parser.add_argument("--min_patches_for_pca", type=int, default=50)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) build metric
    metric = WeakTexturePCANoise(
        patch_size=args.patch_size,
        stride=args.stride,
        weak_texture_percentile=args.weak_texture_percentile,
        max_patches=args.max_patches,
        tail_eig_k=args.tail_eig_k,
        min_patches_for_pca=args.min_patches_for_pca,
    )

    # 2) list images (optionally subsample)
    paths = list_images(args.img_dir)
    if len(paths) == 0:
        raise RuntimeError(f"No images found in {args.img_dir}")

    if args.max_n > 0 and len(paths) > args.max_n:
        # 固定随机种子保证可复现
        rng = np.random.default_rng(0)
        paths = rng.choice(paths, size=args.max_n, replace=False).tolist()

    print(f"[INFO] Scoring {len(paths)} images...")

    # 3) score
    scored = []
    for i, p in enumerate(paths):
        try:
            img = read_bgr(p)
            s = float(metric.get_score(img, mask=None))
            scored.append((p, s))
        except Exception as e:
            # 读图/计算失败直接跳过
            continue

        if (i + 1) % 200 == 0:
            print(f"[INFO] {i+1}/{len(paths)} done")

    if len(scored) == 0:
        raise RuntimeError("No scored images (all failed?).")

    # 4) sort
    scored_sorted = sorted(scored, key=lambda x: x[1])
    low = scored_sorted[: args.bottomk]
    high = list(reversed(scored_sorted[-args.topk:]))

    # 5) export
    low_dir = os.path.join(args.out_dir, "lowest")
    high_dir = os.path.join(args.out_dir, "highest")

    low_exported = export_ranked_images(low, low_dir, k=len(low), prefix="LOW")
    high_exported = export_ranked_images(high, high_dir, k=len(high), prefix="HIGH")

    # 6) save grids
    save_grid(low_exported, os.path.join(args.out_dir, "grid_lowest.jpg"), cols=10, thumb=192)
    save_grid(high_exported, os.path.join(args.out_dir, "grid_highest.jpg"), cols=10, thumb=192)

    # 7) save csv for analysis
    csv_path = os.path.join(args.out_dir, "scores.csv")
    with open(csv_path, "w") as f:
        f.write("path,noise_pca_sigma\n")
        for p, s in scored_sorted:
            f.write(f"{p},{s}\n")

    # 8) quick stats
    sigmas = np.array([s for _, s in scored_sorted], dtype=np.float32)
    print("[DONE]")
    print(f"saved: {args.out_dir}")
    print(f"count={len(sigmas)}, min={sigmas.min():.6f}, median={np.median(sigmas):.6f}, p90={np.quantile(sigmas,0.9):.6f}, max={sigmas.max():.6f}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()