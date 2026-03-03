import os, json
import numpy as np

def load_pair(use_200=True):
    emb = "visual_emb_200.npy" if use_200 and os.path.exists("visual_emb_200.npy") else "visual_emb.npy"
    paths = "clip_paths_200_abs.json" if use_200 and os.path.exists("clip_paths_200_abs.json") else "clip_paths.json"
    Z = np.load(emb)
    with open(paths, "r") as f:
        P = json.load(f)
    return Z, P, emb, paths

def main():
    Z, P, embf, pathf = load_pair(use_200=True)
    print(f"[E] loaded: {embf} shape={Z.shape}")
    print(f"[E] loaded: {pathf} n={len(P)}")

    path2idx = {p:i for i,p in enumerate(P)}
    miss = 0
    for p in P[:50]:
        if path2idx.get(p) is None:
            miss += 1
    print(f"[E] quick lookup self-check on first 50: miss={miss}")

    # also check duplicates
    uniq = len(set(P))
    print(f"[E] unique paths: {uniq}/{len(P)} duplicates={len(P)-uniq}")

    # check that all paths exist on disk (optional)
    not_exist = [p for p in P if not os.path.exists(p)]
    print(f"[E] paths not found on disk: {len(not_exist)}/{len(P)}")
    if len(not_exist) > 0:
        print("  examples:", not_exist[:5])

if __name__ == "__main__":
    main()