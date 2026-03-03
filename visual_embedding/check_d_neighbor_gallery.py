import os, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import faiss
except ImportError:
    raise SystemExit("FAISS not found. Install: conda install -c conda-forge faiss-cpu")

EPS = 1e-12

def l2_normalize(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + EPS)

def load_pair(use_200=True):
    emb = "visual_emb_200.npy" if use_200 and os.path.exists("visual_emb_200.npy") else "visual_emb.npy"
    paths = "clip_paths_200_abs.json" if use_200 and os.path.exists("clip_paths_200_abs.json") else "clip_paths.json"
    Z = np.load(emb).astype(np.float32)
    with open(paths, "r") as f:
        P = json.load(f)
    assert len(P) == Z.shape[0]
    return Z, P

def knn(Z, idx, k=10):
    Z = l2_normalize(Z)
    index = faiss.IndexFlatIP(Z.shape[1])
    index.add(Z)
    q = Z[idx:idx+1]
    sims, nbrs = index.search(q, k+1)  # include self then drop
    nbrs = nbrs[0].tolist()
    sims = sims[0].tolist()
    # drop self
    out = [(j, s) for j, s in zip(nbrs, sims) if j != idx]
    return out[:k]

def make_grid(paths, labels, out_png, cell=224, cols=6):
    rows = int(np.ceil(len(paths)/cols))
    W = cols*cell
    H = rows*cell
    canvas = Image.new("RGB", (W, H), (255,255,255))

    for i, (p, lab) in enumerate(zip(paths, labels)):
        r = i // cols
        c = i % cols
        x0, y0 = c*cell, r*cell
        try:
            im = Image.open(p).convert("RGB")
            im = im.resize((cell, cell))
        except Exception:
            im = Image.new("RGB", (cell, cell), (200, 200, 200))
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, 0, cell, 24], fill=(0,0,0))
        draw.text((4, 4), lab, fill=(255,255,255))
        canvas.paste(im, (x0, y0))

    canvas.save(out_png)
    print("saved", out_png)

def compute_inv_mean_dist_all(Z, k=50):
    Z = l2_normalize(Z)
    index = faiss.IndexFlatIP(Z.shape[1])
    index.add(Z)
    topk = min(Z.shape[0], k+1)
    sims, nbrs = index.search(Z, topk)
    inv = np.zeros((Z.shape[0],), np.float32)
    for i in range(Z.shape[0]):
        s = sims[i]
        ids = nbrs[i]
        keep = ids != i
        s = s[keep][:k]
        d = 1.0 - s
        md = float(np.mean(d)) if d.size else 0.0
        inv[i] = float(1.0/(md+EPS)) if md > 0 else 0.0
    return inv

def main():
    Z, P = load_pair(use_200=True)
    inv = compute_inv_mean_dist_all(Z, k=50)

    dense_idx = int(np.argmax(inv))
    sparse_idx = int(np.argmin(inv))

    # dense gallery: query + top10
    dense_neighbors = knn(Z, dense_idx, k=10)
    dense_paths = [P[dense_idx]] + [P[j] for j,_ in dense_neighbors]
    dense_labels = [f"Q inv={inv[dense_idx]:.2f}"] + [f"{t+1} sim={s:.2f}" for t,(j,s) in enumerate(dense_neighbors)]
    make_grid(dense_paths, dense_labels, "gallery_dense.png")

    # sparse gallery
    sparse_neighbors = knn(Z, sparse_idx, k=10)
    sparse_paths = [P[sparse_idx]] + [P[j] for j,_ in sparse_neighbors]
    sparse_labels = [f"Q inv={inv[sparse_idx]:.2f}"] + [f"{t+1} sim={s:.2f}" for t,(j,s) in enumerate(sparse_neighbors)]
    make_grid(sparse_paths, sparse_labels, "gallery_sparse.png")

    print(f"dense_idx={dense_idx} path={P[dense_idx]}")
    print(f"sparse_idx={sparse_idx} path={P[sparse_idx]}")

if __name__ == "__main__":
    main()