import argparse
import json
import os
import random
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm


def parse_sizes(s: str) -> List[int]:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    out = [int(v) for v in vals]
    if any(v <= 0 for v in out):
        raise ValueError("All sizes must be positive integers.")
    return out


def stable_sample_images(img_dir: str, n: int, seed: int) -> List[str]:
    paths = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if not paths:
        return []
    rng = random.Random(seed)
    return rng.sample(paths, min(n, len(paths)))


def compute_vector_stats(vectors: np.ndarray) -> Dict[str, object]:
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D [N, D].")

    l1 = vectors.sum(axis=1)
    nonzero_ratio = (vectors > 0).mean(axis=1)
    return {
        "num_samples": int(vectors.shape[0]),
        "vector_dim": int(vectors.shape[1]),
        "l1_sum_mean": float(np.mean(l1)),
        "l1_sum_std": float(np.std(l1)),
        "nonzero_ratio_mean": float(np.mean(nonzero_ratio)),
        "nonzero_ratio_std": float(np.std(nonzero_ratio)),
        "per_dim_mean": vectors.mean(axis=0).astype(np.float32).tolist(),
        "per_dim_std": vectors.std(axis=0).astype(np.float32).tolist(),
        "per_dim_min": vectors.min(axis=0).astype(np.float32).tolist(),
        "per_dim_max": vectors.max(axis=0).astype(np.float32).tolist(),
    }


def pca_2d(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X must be 2D [N, D].")
    if X.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    Y = U[:, :2] * S[:2]
    if Y.shape[1] < 2:
        Y = np.pad(Y, ((0, 0), (0, 2 - Y.shape[1])), mode="constant")
    return Y.astype(np.float32)


def save_dim_histograms(vectors: np.ndarray, out_png: str, bins: int = 20) -> None:
    import matplotlib.pyplot as plt

    D = vectors.shape[1]
    ncols = 4
    nrows = int(np.ceil(D / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for d in range(nrows * ncols):
        ax = axes[d // ncols, d % ncols]
        if d < D:
            ax.hist(vectors[:, d], bins=bins, color="steelblue", alpha=0.85)
            ax.set_title(f"dim {d}")
        else:
            ax.axis("off")

    fig.suptitle("Laplacian Vector Distribution per Dimension")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_pca_scatter(vectors: np.ndarray, out_png: str) -> None:
    import matplotlib.pyplot as plt

    Y = pca_2d(vectors)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Y[:, 0], Y[:, 1], s=8, alpha=0.6, color="darkorange")
    ax.set_title("PCA (2D) of Laplacian Vectors")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def extract_vectors(
    img_paths: Sequence[str],
    vector_meta: Dict[str, object],
    show_progress: bool = True,
) -> Tuple[List[str], np.ndarray]:
    import cv2
    from clip_dinoiser.feature_utils.data_feature.implementations.quality import LaplacianSharpness

    metric = LaplacianSharpness()
    kept_paths: List[str] = []
    vectors: List[np.ndarray] = []

    iterator = tqdm(
        img_paths,
        desc="Extracting vectors",
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for p in iterator:
        img = cv2.imread(p)
        if img is None:
            continue
        try:
            vec = metric.get_vector_score(img, meta=vector_meta)
        except Exception:
            continue
        kept_paths.append(p)
        vectors.append(vec.astype(np.float32))

    if not vectors:
        return kept_paths, np.zeros((0, int(vector_meta.get("num_bins", 16))), dtype=np.float32)

    return kept_paths, np.stack(vectors, axis=0)


def run_for_scale(
    img_dir: str,
    size: int,
    seed: int,
    out_root: str,
    vector_meta: Dict[str, object],
    sample_dump_k: int,
    show_progress: bool,
) -> None:
    t0 = time.time()
    print(f"[Sanity] Start n={size} (seed={seed})")
    sample_paths = stable_sample_images(img_dir, size, seed)
    print(f"[Sanity] Sampled {len(sample_paths)} image paths from: {img_dir}")
    kept_paths, vectors = extract_vectors(sample_paths, vector_meta, show_progress=show_progress)
    print(f"[Sanity] Extracted vectors for {vectors.shape[0]} / {len(sample_paths)} samples")

    out_dir = os.path.join(out_root, f"n_{size}")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "vectors.npy"), vectors)
    with open(os.path.join(out_dir, "paths.json"), "w") as f:
        json.dump(kept_paths, f, indent=2)

    if vectors.shape[0] == 0:
        with open(os.path.join(out_dir, "vector_stats.json"), "w") as f:
            json.dump({"num_samples": 0, "message": "No valid vectors extracted."}, f, indent=2)
        print(f"[Sanity] n={size} no valid vectors. Outputs in: {out_dir}")
        return

    stats = compute_vector_stats(vectors)
    stats["requested_size"] = int(size)
    stats["kept_size"] = int(vectors.shape[0])
    stats["vector_meta"] = vector_meta

    with open(os.path.join(out_dir, "vector_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # dump single-sample vectors for manual inspection
    dump_idx = list(range(min(sample_dump_k, vectors.shape[0])))
    with open(os.path.join(out_dir, "sample_vectors.jsonl"), "w") as f:
        for i in dump_idx:
            row = {
                "index": int(i),
                "path": kept_paths[i],
                "vector": vectors[i].tolist(),
                "l1_sum": float(vectors[i].sum()),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    save_dim_histograms(vectors, os.path.join(out_dir, "hist_per_dim.png"))
    save_pca_scatter(vectors, os.path.join(out_dir, "pca_scatter.png"))
    dt = time.time() - t0
    print(f"[Sanity] Done n={size} in {dt:.2f}s. Outputs in: {out_dir}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sanity check Laplacian vector features (no training).")
    p.add_argument("--img-dir", default="./data/coco_stuff164k/images/train2017")
    p.add_argument("--sizes", default="200,2000,20000", help="comma-separated sample sizes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="./clip_dinoiser/sanity/laplacian_vector")

    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--num-bins", type=int, default=16)
    p.add_argument("--use-log", action="store_true", default=True)
    p.add_argument("--no-use-log", action="store_false", dest="use_log")
    p.add_argument("--l1-normalize", action="store_true", default=True)
    p.add_argument("--no-l1-normalize", action="store_false", dest="l1_normalize")

    p.add_argument("--sample-dump-k", type=int, default=10)
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    sizes = parse_sizes(args.sizes)

    if not os.path.isdir(args.img_dir):
        raise FileNotFoundError(f"Image directory not found: {args.img_dir}")

    vector_meta: Dict[str, object] = {
        "patch_size": args.patch_size,
        "stride": args.stride,
        "num_bins": args.num_bins,
        "use_log": args.use_log,
        "l1_normalize": args.l1_normalize,
    }

    os.makedirs(args.out_dir, exist_ok=True)

    print("[Sanity] Laplacian vector sanity-check")
    print(f"[Sanity] sizes={sizes}, out_dir={args.out_dir}")
    progress_bar = tqdm(
        sizes,
        desc="Scales",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    for n in progress_bar:
        run_for_scale(
            img_dir=args.img_dir,
            size=n,
            seed=args.seed,
            out_root=args.out_dir,
            vector_meta=vector_meta,
            sample_dump_k=args.sample_dump_k,
            show_progress=not args.no_progress,
        )

    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as f:
        json.dump(
            {
                "img_dir": args.img_dir,
                "sizes": sizes,
                "seed": args.seed,
                "vector_meta": vector_meta,
            },
            f,
            indent=2,
        )
    print(f"[Sanity] All done. Run meta saved to: {os.path.join(args.out_dir, 'run_meta.json')}")


if __name__ == "__main__":
    main()
