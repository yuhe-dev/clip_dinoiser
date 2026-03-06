# visual_embedding/train_prototypes_kmeans.py
import os, json, argparse
import numpy as np

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="visual_emb.npy")
    ap.add_argument("--out_prefix", default="prototypes")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--niter", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_gpu", action="store_true")
    args = ap.parse_args()

    Z = np.load(args.emb)
    if Z.dtype != np.float32:
        Z = Z.astype(np.float32)

    # cosine/IP requires unit norm
    Z = l2_normalize(Z)

    try:
        import faiss  # type: ignore
    except ImportError:
        raise RuntimeError("Need faiss. Install: conda install -c conda-forge faiss-cpu (or faiss-gpu)")

    d = Z.shape[1]
    k = args.k

    # FAISS KMeans (IP on normalized vectors ~= cosine similarity)
    km = faiss.Kmeans(d, k, niter=args.niter, seed=args.seed, verbose=True, gpu=args.use_gpu)
    km.train(Z)

    C = km.centroids  # [K, D], float32
    C = l2_normalize(C)

    out_npy = f"{args.out_prefix}_k{k}.npy"
    out_meta = f"{args.out_prefix}_meta_k{k}.json"
    np.save(out_npy, C)

    with open(out_meta, "w") as f:
        json.dump({
            "emb": args.emb,
            "k": k,
            "niter": args.niter,
            "seed": args.seed,
            "use_gpu": bool(args.use_gpu),
            "note": "centroids are L2-normalized; use IndexFlatIP for cosine similarity"
        }, f, indent=2)

    print("saved:", out_npy, C.shape)
    print("saved:", out_meta)

if __name__ == "__main__":
    main()