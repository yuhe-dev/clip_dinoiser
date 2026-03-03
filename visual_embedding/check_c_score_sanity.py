import os, json
import numpy as np

try:
    import faiss
except ImportError:
    raise SystemExit("FAISS not found. Install: conda install -c conda-forge faiss-cpu")

EPS = 1e-12

def l2_normalize(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + EPS)

def load_pair(use_200=True):
    emb = "visual_emb_200.npy" if use_200 and os.path.exists("visual_emb_200.npy") else "visual_emb.npy"
    paths = "clip_paths_200.json" if use_200 and os.path.exists("clip_paths_200.json") else "clip_paths.json"
    Z = np.load(emb).astype(np.float32)
    with open(paths, "r") as f:
        P = json.load(f)
    assert len(P) == Z.shape[0]
    return Z, P, emb, paths

def compute_inv_mean_dist(Z, k=50, include_self=False):
    Z = l2_normalize(Z)
    index = faiss.IndexFlatIP(Z.shape[1])
    index.add(Z)

    topk = min(Z.shape[0], k + (0 if include_self else 1))
    scores, nbrs = index.search(Z, topk)  # query all

    invs = np.zeros((Z.shape[0],), np.float32)
    mean_dists = np.zeros((Z.shape[0],), np.float32)

    for i in range(Z.shape[0]):
        sims = scores[i]
        ids = nbrs[i]
        if not include_self:
            keep = ids != i
            sims = sims[keep][:k]
        else:
            sims = sims[:k]
        dists = 1.0 - sims
        md = float(np.mean(dists)) if dists.size else 0.0
        mean_dists[i] = md
        invs[i] = float(1.0 / (md + EPS)) if md > 0 else 0.0

    return invs, mean_dists

def main(k=50):
    Z, P, embf, pathf = load_pair(use_200=True)
    print(f"[C] loaded: {embf} {Z.shape}, {pathf} {len(P)}")
    invs, mds = compute_inv_mean_dist(Z, k=k, include_self=False)

    print(f"[C] mean_dist stats: min={mds.min():.4f} p50={np.median(mds):.4f} p95={np.percentile(mds,95):.4f} max={mds.max():.4f}")
    print(f"[C] inv_mean_dist stats: min={invs.min():.2f} p50={np.median(invs):.2f} p95={np.percentile(invs,95):.2f} max={invs.max():.2f}")

    # show top/bottom
    top = np.argsort(-invs)[:5]
    bot = np.argsort(invs)[:5]
    print("\nTop-5 densest (highest inv_mean_dist):")
    for i in top:
        print(f"  score={invs[i]:.3f} mean_dist={mds[i]:.4f} path={P[i]}")
    print("\nBottom-5 sparsest (lowest inv_mean_dist):")
    for i in bot:
        print(f"  score={invs[i]:.3f} mean_dist={mds[i]:.4f} path={P[i]}")

if __name__ == "__main__":
    main()