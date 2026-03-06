import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

EPS = 1e-12


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + EPS)


def load_embeddings_and_paths(emb_file: str, paths_file: str) -> Tuple[np.ndarray, List[str]]:
    z = np.load(emb_file).astype(np.float32)
    with open(paths_file, "r") as f:
        p = json.load(f)
    if len(p) != z.shape[0]:
        raise ValueError(f"len(paths)={len(p)} != emb_rows={z.shape[0]}")
    return z, p


def build_path2idx(paths: List[str]) -> Dict[str, int]:
    path2idx: Dict[str, int] = {}
    for i, p in enumerate(paths):
        path2idx.setdefault(p, i)
        ap = os.path.abspath(p)
        path2idx.setdefault(ap, i)
        try:
            rp = os.path.relpath(ap)
            path2idx.setdefault(rp, i)
        except Exception:
            pass
        path2idx.setdefault(os.path.basename(p), i)
    return path2idx


def lookup_idx(path2idx: Dict[str, int], qpath: str) -> Optional[int]:
    cands = [qpath, os.path.abspath(qpath), os.path.basename(qpath)]
    try:
        cands.append(os.path.relpath(os.path.abspath(qpath)))
    except Exception:
        pass
    for k in cands:
        if k in path2idx:
            return path2idx[k]
    return None


def summarize_array(x: np.ndarray, name: str) -> None:
    p = [1, 5, 25, 50, 75, 95, 99]
    vals = [float(np.percentile(x, t)) for t in p]
    parts = " ".join([f"p{t}={v:.6f}" for t, v in zip(p, vals)])
    print(f"[{name}] min={float(x.min()):.6f} max={float(x.max()):.6f} mean={float(x.mean()):.6f} {parts}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="visual_embedding")
    ap.add_argument("--emb_file", default="visual_emb.npy")
    ap.add_argument("--paths_file", default="clip_paths_abs.json")
    ap.add_argument("--centroid_file", default="prototypes_k20.npy")
    ap.add_argument("--check_n", type=int, default=1000, help="path-hit sample count")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    emb_path = os.path.join(args.cache_dir, args.emb_file)
    paths_path = os.path.join(args.cache_dir, args.paths_file)
    cen_path = os.path.join(args.cache_dir, args.centroid_file)

    if not os.path.exists(emb_path):
        raise FileNotFoundError(emb_path)
    if not os.path.exists(paths_path):
        raise FileNotFoundError(paths_path)
    if not os.path.exists(cen_path):
        raise FileNotFoundError(cen_path)

    z, paths = load_embeddings_and_paths(emb_path, paths_path)
    c = np.load(cen_path).astype(np.float32)
    if z.shape[1] != c.shape[1]:
        raise ValueError(f"dim mismatch: emb D={z.shape[1]} vs centroid D={c.shape[1]}")

    z = l2_normalize(z)
    c = l2_normalize(c)
    c_norm = np.linalg.norm(c, axis=1)
    print(f"[meta] emb={emb_path} shape={z.shape}")
    print(f"[meta] paths={paths_path} n={len(paths)}")
    print(f"[meta] centroids={cen_path} shape={c.shape}")
    print(f"[meta] centroid_norm min/mean/max: {float(c_norm.min()):.6f}/{float(c_norm.mean()):.6f}/{float(c_norm.max()):.6f}")

    # 1) Path match rate sanity
    path2idx = build_path2idx(paths)
    rng = random.Random(args.seed)
    n = min(args.check_n, len(paths))
    sampled = rng.sample(paths, n)
    hit = sum(1 for p in sampled if lookup_idx(path2idx, p) is not None)
    print(f"[path_match] sampled={n} hit={hit} miss={n-hit} hit_rate={hit/max(n,1):.4f}")

    # 2) Margin distribution (all rows)
    sims = z @ c.T  # cosine sim because normalized
    top2 = np.partition(sims, -2, axis=1)[:, -2:]
    s1 = np.maximum(top2[:, 0], top2[:, 1])  # nearest centroid sim
    s2 = np.minimum(top2[:, 0], top2[:, 1])  # 2nd nearest centroid sim
    d1 = 1.0 - s1
    d2 = 1.0 - s2
    margin = d2 - d1

    summarize_array(d1, "d1_nearest_dist")
    summarize_array(margin, "prototype_margin")
    nonzero = float(np.mean(np.abs(margin) > 1e-9))
    print(f"[margin] nonzero_ratio={nonzero:.4f}")

    # 3) Cluster assignment balance
    assign = np.argmax(sims, axis=1)
    counts = np.bincount(assign, minlength=c.shape[0])
    empty = int((counts == 0).sum())
    p = counts / max(counts.sum(), 1)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    ent_max = float(np.log(len(p)))
    top5_ratio = float(np.sort(p)[-5:].sum()) if len(p) >= 5 else float(np.sort(p).sum())
    print(f"[cluster] K={len(counts)} empty={empty} min={int(counts.min())} p50={int(np.median(counts))} max={int(counts.max())}")
    print(f"[cluster] entropy={ent:.6f} entropy_max={ent_max:.6f} normalized={ent/max(ent_max,1e-12):.6f}")
    print(f"[cluster] top5_ratio={top5_ratio:.6f}")


if __name__ == "__main__":
    main()
