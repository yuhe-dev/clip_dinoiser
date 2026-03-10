import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np


def l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / (norms + eps)


def save_prototype_bundle(
    output_root: str,
    centroids: np.ndarray,
    prototype_meta: Dict[str, object],
    embeddings_file: str,
    paths_file: str,
) -> Tuple[str, str]:
    os.makedirs(output_root, exist_ok=True)

    k = int(prototype_meta.get("k", centroids.shape[0]))
    centroid_path = os.path.join(output_root, f"prototypes_k{k}.npy")
    config_path = os.path.join(output_root, f"prototypes_meta_k{k}.json")

    np.save(centroid_path, np.asarray(centroids, dtype=np.float32))
    config = {
        "prototype_meta": prototype_meta,
        "embeddings_file": embeddings_file,
        "paths_file": paths_file,
        "centroid_file": os.path.basename(centroid_path),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return centroid_path, config_path


def load_embedding_matrix(embedding_root: str, embeddings_file: str) -> np.ndarray:
    embedding_path = os.path.join(embedding_root, embeddings_file)
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    return np.load(embedding_path).astype(np.float32)


def build_prototypes(
    embeddings: np.ndarray,
    k: int,
    niter: int,
    seed: int,
    use_gpu: bool,
) -> np.ndarray:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty [N, D] matrix")

    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise ImportError("FAISS is required to build coverage prototypes.") from exc

    normalized = l2_normalize(embeddings.astype(np.float32))
    k = min(int(k), normalized.shape[0])
    if k <= 0:
        raise ValueError("k must be > 0")

    d = int(normalized.shape[1])
    km = faiss.Kmeans(d, k, niter=int(niter), seed=int(seed), verbose=True, gpu=bool(use_gpu))
    km.train(normalized)
    return l2_normalize(np.asarray(km.centroids, dtype=np.float32))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build coverage prototypes from saved CLIP embeddings.")
    parser.add_argument("--embedding-root", default="./data/data_feature/coverage/visual_embedding")
    parser.add_argument("--embeddings-file", default="visual_emb.npy")
    parser.add_argument("--paths-file", default="clip_paths_abs.json")
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--niter", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-gpu", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    embedding_root = os.path.abspath(args.embedding_root)
    embeddings = load_embedding_matrix(embedding_root, args.embeddings_file)
    prototype_meta = {
        "k": int(args.k),
        "niter": int(args.niter),
        "seed": int(args.seed),
        "use_gpu": bool(args.use_gpu),
    }

    print(f"[coverage-prototype] embedding_root={embedding_root}")
    print(f"[coverage-prototype] embeddings_shape={embeddings.shape}")
    print(f"[coverage-prototype] prototype_meta={prototype_meta}")

    centroids = build_prototypes(
        embeddings=embeddings,
        k=args.k,
        niter=args.niter,
        seed=args.seed,
        use_gpu=args.use_gpu,
    )
    centroid_path, config_path = save_prototype_bundle(
        output_root=embedding_root,
        centroids=centroids,
        prototype_meta=prototype_meta,
        embeddings_file=args.embeddings_file,
        paths_file=args.paths_file,
    )

    print(f"[coverage-prototype] saved centroids: {centroid_path}")
    print(f"[coverage-prototype] saved config: {config_path}")
    print(f"[coverage-prototype] centroid_shape={centroids.shape}")


if __name__ == "__main__":
    main()
