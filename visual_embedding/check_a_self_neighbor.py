import os, json
import numpy as np

try:
    import faiss  # pip/conda install faiss-cpu
except ImportError:
    raise SystemExit("FAISS not found. Install: conda install -c conda-forge faiss-cpu")

EPS = 1e-12

def l2_normalize(Z: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / (n + EPS)

def load_pair(prefix_200=True):
    emb = "visual_emb_200.npy" if prefix_200 and os.path.exists("visual_emb_200.npy") else "visual_emb.npy"
    paths = "clip_paths_200.json" if prefix_200 and os.path.exists("clip_paths_200.json") else "clip_paths.json"
    Z = np.load(emb).astype(np.float32)
    with open(paths, "r") as f:
        P = json.load(f)
    assert len(P) == Z.shape[0], f"len(paths)={len(P)} != Z.shape[0]={Z.shape[0]}"
    return Z, P, emb, paths

def main(n_checks=10, metric="cosine"):
    Z, P, embf, pathf = load_pair(prefix_200=True)
    print(f"[A] loaded: {embf} {Z.shape}, {pathf} {len(P)}")

    if metric == "cosine":
        Z = l2_normalize(Z)
        index = faiss.IndexFlatIP(Z.shape[1])
    else:
        index = faiss.IndexFlatL2(Z.shape[1])

    index.add(Z)

    ok = 0
    for idx in range(min(n_checks, Z.shape[0])):
        q = Z[idx:idx+1]
        scores, nbrs = index.search(q, 1)  # include_self => top1 should be itself
        top_idx = int(nbrs[0, 0])
        top_score = float(scores[0, 0])
        if metric == "cosine":
            # IP similarity ~ 1
            good = (top_idx == idx) and (top_score > 0.999)
        else:
            # L2 distance squared ~ 0
            good = (top_idx == idx) and (top_score < 1e-6)

        print(f"idx={idx} top_idx={top_idx} top_score={top_score:.6f} path={os.path.basename(P[idx])} good={good}")
        ok += int(good)

    print(f"[A] pass {ok}/{min(n_checks, Z.shape[0])}")

if __name__ == "__main__":
    main()