import os
import json
import numpy as np
from typing import Optional, Dict, Any

from ..dimensions import CoverageDimension

try:
    import faiss  # type: ignore
except ImportError as e:
    raise ImportError(
        "FAISS not installed. Install with: pip install faiss-cpu (or faiss-gpu)"
    ) from e


class KNNLocalDensityCLIPFaiss(CoverageDimension):
    """
    Local density (kNN-based) in a precomputed CLIP visual embedding space using FAISS.

    Cache directory must contain:
      - visual_emb.npy  : float32 [N, D] embeddings (recommended: L2-normalized if using cosine)
      - clip_paths.json : list[str] length N

    get_score expects meta contains:
      - meta['img_path'] or meta['path'] matching an entry in clip_paths.json

    metric:
      - "cosine": uses inner product on unit-normalized vectors (FAISS IndexFlatIP)
      - "l2": uses Euclidean distance (FAISS IndexFlatL2)

    mode:
      - "mean_dist": mean neighbor distance (smaller => denser)
      - "inv_mean_dist": 1/(eps + mean_dist) (larger => denser)
      - "radius_count": count neighbors within radius (larger => denser)
          * for cosine: radius is cosine distance threshold, i.e. (1 - sim) <= radius
          * for l2: radius is L2 distance threshold
    """

    def __init__(
        self,
        cache_dir: str = "visual_embedding",
        emb_file: str = "visual_emb.npy",
        paths_file: str = "clip_paths_abs.json",
        k: int = 50,
        metric: str = "cosine",        # "cosine" or "l2"
        mode: str = "inv_mean_dist",   # "mean_dist" | "inv_mean_dist" | "radius_count"
        radius: float = 0.2,           # for radius_count
        eps: float = 1e-12,
        include_self: bool = False,
        normalize_for_cosine: bool = True,  # ensure embeddings are unit norm for cosine
        use_gpu: bool = False,              # optional FAISS GPU
        gpu_id: int = 0,
    ):
        super().__init__("knn_local_density_faiss")
        self.cache_dir = cache_dir
        self.emb_path = os.path.join(cache_dir, emb_file)
        self.paths_path = os.path.join(cache_dir, paths_file)

        self.k = int(k)
        self.metric = metric.lower().strip()
        self.mode = mode.lower().strip()
        self.radius = float(radius)
        self.eps = float(eps)
        self.include_self = bool(include_self)
        self.normalize_for_cosine = bool(normalize_for_cosine)
        self.use_gpu = bool(use_gpu)
        self.gpu_id = int(gpu_id)

        self._Z: Optional[np.ndarray] = None
        self._paths: Optional[list] = None
        self._path2idx: Optional[Dict[str, int]] = None
        self._index = None
        self._D: Optional[int] = None

        if self.k <= 0:
            raise ValueError("k must be > 0")
        if self.metric not in {"cosine", "l2"}:
            raise ValueError("metric must be 'cosine' or 'l2'")
        if self.mode not in {"mean_dist", "inv_mean_dist", "radius_count"}:
            raise ValueError("mode must be one of: mean_dist, inv_mean_dist, radius_count")

    @staticmethod
    def _l2_normalize(Z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / (n + eps)

    def _lazy_init(self):
        if self._index is not None:
            return

        if not os.path.exists(self.emb_path):
            raise FileNotFoundError(f"Embedding file not found: {self.emb_path}")
        if not os.path.exists(self.paths_path):
            raise FileNotFoundError(f"Paths file not found: {self.paths_path}")

        Z = np.load(self.emb_path)
        if Z.dtype != np.float32:
            Z = Z.astype(np.float32)

        with open(self.paths_path, "r") as f:
            paths = json.load(f)

        if len(paths) != Z.shape[0]:
            raise ValueError(f"Mismatch: len(paths)={len(paths)} vs Z.shape[0]={Z.shape[0]}")

        # For cosine/IP, embeddings should be unit norm
        if self.metric == "cosine" and self.normalize_for_cosine:
            Z = self._l2_normalize(Z, eps=self.eps)

        self._Z = Z
        self._paths = paths
        self._path2idx = {}
        for i, p in enumerate(paths):
            self._path2idx[p] = i
            ap = os.path.abspath(p)
            self._path2idx.setdefault(ap, i)  # also allow absolute path lookup
            rp = os.path.relpath(p)
            self._path2idx.setdefault(rp, i)  # also allow relative path lookup
        self._D = int(Z.shape[1])

        # Build FAISS index
        if self.metric == "cosine":
            index = faiss.IndexFlatIP(self._D)   # inner product
        else:
            index = faiss.IndexFlatL2(self._D)   # squared L2 distance

        index.add(Z)  # type: ignore

        # Optional GPU
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, self.gpu_id, index)

        self._index = index

    def _lookup_index(self, meta: Dict[str, Any]) -> Optional[int]:
        if self._path2idx is None:
            return None
        p = meta.get("img_path") or meta.get("path")
        if p is None:
            return None
        cands = [p, os.path.abspath(p), os.path.relpath(os.path.abspath(p))]
        for q in cands:
            if q in self._path2idx:
                return self._path2idx[q]
        return None

    def _search(self, q: np.ndarray, topk: int):
        """
        Returns (scores, indices)
        - cosine metric: scores are inner products (similarities)
        - l2 metric: scores are squared L2 distances
        """
        assert self._index is not None
        scores, idxs = self._index.search(q, topk)  # type: ignore
        return scores[0], idxs[0]

    def get_score(self, image, mask=None, meta=None) -> float:
        """
        image/mask are unused; we use meta['img_path'] to find embedding row.
        """
        meta = meta or {}
        self._lazy_init()

        idx = self._lookup_index(meta)
        if idx is None:
            return 0.0

        assert self._Z is not None

        q = self._Z[idx:idx+1].astype(np.float32)
        if self.metric == "cosine" and self.normalize_for_cosine:
            q = self._l2_normalize(q, eps=self.eps)

        # request extra neighbor if we might drop self
        topk = min(self.k + (1 if not self.include_self else 0), self._Z.shape[0])
        scores, nbrs = self._search(q, topk=topk)

        # drop invalid (-1)
        valid = nbrs >= 0
        scores = scores[valid]
        nbrs = nbrs[valid]

        if not self.include_self:
            keep = nbrs != idx
            scores = scores[keep]
            nbrs = nbrs[keep]

        if scores.size == 0:
            return 0.0

        scores = scores[: self.k]

        # convert FAISS output -> distance array dists
        if self.metric == "cosine":
            # scores = inner product similarity in [-1, 1] (for normalized vectors)
            sims = scores
            dists = 1.0 - sims  # cosine distance
        else:
            # scores = squared L2 distance
            dists = np.sqrt(np.maximum(scores, 0.0))

        if self.mode == "mean_dist":
            return float(np.mean(dists))
        elif self.mode == "inv_mean_dist":
            return float(1.0 / (self.eps + np.mean(dists)))
        else:  # radius_count
            return float(np.sum(dists <= self.radius))
        


class PrototypeMarginCLIPFaiss(CoverageDimension):
    """
    Global position via prototype margin: margin = d2 - d1,
    where d = 1 - cosine_sim(v, centroid). Requires:
      - visual embeddings: [N,D] (visual_emb.npy)
      - paths list: clip_paths_abs.json (or clip_paths.json)
      - centroids: prototypes_k{K}.npy [K,D]
    """

    # IMPORTANT: sampler can skip reading image/mask for this metric
    needs_image = False
    needs_mask = False

    def __init__(
        self,
        cache_dir: str = "visual_embedding",
        emb_file: str = "visual_emb.npy",
        paths_file: str = "clip_paths_abs.json",
        centroid_file: str = "prototypes_k200.npy",
        normalize: bool = True,
        eps: float = 1e-12,
        use_gpu: bool = False,
        gpu_id: int = 0,
    ):
        super().__init__("prototype_margin_faiss")
        self.cache_dir = cache_dir
        self.emb_path = os.path.join(cache_dir, emb_file)
        self.paths_path = os.path.join(cache_dir, paths_file)
        self.centroid_path = os.path.join(cache_dir, centroid_file)

        self.normalize = bool(normalize)
        self.eps = float(eps)
        self.use_gpu = bool(use_gpu)
        self.gpu_id = int(gpu_id)

        self._Z: Optional[np.ndarray] = None
        self._C: Optional[np.ndarray] = None
        self._path2idx: Optional[Dict[str, int]] = None
        self._centroid_index = None
        self._D: Optional[int] = None

    @staticmethod
    def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (n + eps)

    def _lazy_init(self):
        if self._centroid_index is not None:
            return

        for p in [self.emb_path, self.paths_path, self.centroid_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(p)

        Z = np.load(self.emb_path).astype(np.float32)
        C = np.load(self.centroid_path).astype(np.float32)

        with open(self.paths_path, "r") as f:
            paths = json.load(f)

        if len(paths) != Z.shape[0]:
            raise ValueError(f"Mismatch: paths={len(paths)} vs Z={Z.shape[0]}")

        if self.normalize:
            Z = self._l2_normalize(Z, self.eps)
            C = self._l2_normalize(C, self.eps)

        self._Z = Z
        self._C = C
        self._path2idx = {p: i for i, p in enumerate(paths)}
        self._D = int(Z.shape[1])

        # build index over centroids, IP on normalized => cosine sim
        index = faiss.IndexFlatIP(self._D)
        index.add(C)

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, self.gpu_id, index)

        self._centroid_index = index

    def _lookup_idx(self, meta: Dict[str, Any]) -> Optional[int]:
        if self._path2idx is None:
            return None
        p = meta.get("img_path") or meta.get("path")
        if p is None:
            return None
        return self._path2idx.get(p, None)

    def get_score(self, image=None, mask=None, meta=None) -> float:
        meta = meta or {}
        self._lazy_init()

        idx = self._lookup_idx(meta)
        if idx is None:
            return 0.0

        assert self._Z is not None
        assert self._centroid_index is not None

        q = self._Z[idx:idx+1]  # [1,D]
        # search top2 centroids
        sims, cidx = self._centroid_index.search(q, 2)  # sims: [1,2]
        sims = sims[0]
        if sims.shape[0] < 2:
            return 0.0

        # cosine distance
        d1 = 1.0 - float(sims[0])
        d2 = 1.0 - float(sims[1])
        margin = d2 - d1
        return float(margin)