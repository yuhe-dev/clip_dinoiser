import os, json, random
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

def topk_numpy_cosine(Z, idx, k):
    q = Z[idx]
    sims = Z @ q  # [N]
    # argsort descending
    order = np.argsort(-sims)
    return order[:k], sims[order[:k]]

def main(seed=0, n_query=5, k=20):
    Z, P, embf, pathf = load_pair(use_200=True)
    print(f"[B] loaded: {embf} {Z.shape}, {pathf} {len(P)}")

    Z = l2_normalize(Z)
    index = faiss.IndexFlatIP(Z.shape[1])
    index.add(Z)

    rng = random.Random(seed)
    idxs = [rng.randrange(Z.shape[0]) for _ in range(min(n_query, Z.shape[0]))]

    for idx in idxs:
        q = Z[idx:idx+1]
        scores_f, nbrs_f = index.search(q, k)

        nbrs_np, sims_np = topk_numpy_cosine(Z, idx, k)

        nbrs_f = nbrs_f[0]
        sims_f = scores_f[0]

        same = np.all(nbrs_f == nbrs_np)
        max_abs_diff = float(np.max(np.abs(sims_f - sims_np)))
        print(f"\nquery idx={idx} file={os.path.basename(P[idx])}")
        print("faiss top5:", list(zip(nbrs_f[:5].tolist(), np.round(sims_f[:5], 6).tolist())))
        print("numpy top5:", list(zip(nbrs_np[:5].tolist(), np.round(sims_np[:5], 6).tolist())))
        print(f"same_neighbors={same}, max_abs_sim_diff={max_abs_diff:.8f}")

    print("\n[B] If same_neighbors=True and diff ~ 1e-6, FAISS is correct.")

if __name__ == "__main__":
    main()