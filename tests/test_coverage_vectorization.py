import json
import os
import sys
import tempfile
import types
import unittest

import numpy as np


faiss_stub = sys.modules.get("faiss")
if faiss_stub is None:
    faiss_stub = types.SimpleNamespace()
    sys.modules["faiss"] = faiss_stub


class _FlatIndex:
    def __init__(self, dim: int, metric: str):
        self.dim = dim
        self.metric = metric
        self.vectors = np.zeros((0, dim), dtype=np.float32)

    def add(self, values: np.ndarray):
        self.vectors = np.asarray(values, dtype=np.float32)

    def search(self, query: np.ndarray, topk: int):
        query = np.asarray(query, dtype=np.float32)
        if self.metric == "ip":
            scores = query @ self.vectors.T
            order = np.argsort(-scores, axis=1)[:, :topk]
            ranked_scores = np.take_along_axis(scores, order, axis=1)
            return ranked_scores.astype(np.float32), order.astype(np.int64)

        diffs = query[:, None, :] - self.vectors[None, :, :]
        scores = np.sum(diffs * diffs, axis=2)
        order = np.argsort(scores, axis=1)[:, :topk]
        ranked_scores = np.take_along_axis(scores, order, axis=1)
        return ranked_scores.astype(np.float32), order.astype(np.int64)


faiss_stub.IndexFlatIP = getattr(faiss_stub, "IndexFlatIP", lambda dim: _FlatIndex(dim, "ip"))
faiss_stub.IndexFlatL2 = getattr(faiss_stub, "IndexFlatL2", lambda dim: _FlatIndex(dim, "l2"))


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.feature_utils.data_feature.implementations.coverage import (
    KNNLocalDensityCLIPFaiss,
    PrototypeMarginCLIPFaiss,
)


class TestCoverageVectorization(unittest.TestCase):
    def _write_embedding_bundle(self, tmpdir: str):
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )
        paths = [
            "images/train2017/0001.jpg",
            "images/train2017/0002.jpg",
            "images/train2017/0003.jpg",
            "images/train2017/0004.jpg",
        ]
        centroids = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )

        emb_path = os.path.join(tmpdir, "visual_emb.npy")
        paths_path = os.path.join(tmpdir, "clip_paths_abs.json")
        centroid_path = os.path.join(tmpdir, "prototypes_k3.npy")
        np.save(emb_path, embeddings)
        np.save(centroid_path, centroids)
        with open(paths_path, "w", encoding="utf-8") as f:
            json.dump(paths, f)
        return paths

    def test_knn_vector_score_returns_neighbor_distances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._write_embedding_bundle(tmpdir)
            metric = KNNLocalDensityCLIPFaiss(
                cache_dir=tmpdir,
                k=2,
                metric="cosine",
                mode="mean_dist",
            )

            values = metric.get_vector_score(None, meta={"img_path": paths[0]})

            np.testing.assert_allclose(values, np.asarray([0.2, 1.0], dtype=np.float32), atol=1e-6)
            self.assertEqual(values.dtype, np.float32)
            self.assertEqual(values.shape, (2,))
            self.assertAlmostEqual(metric.get_score(None, meta={"img_path": paths[0]}), 0.6, places=6)

    def test_prototype_vector_score_returns_top_m_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._write_embedding_bundle(tmpdir)
            metric = PrototypeMarginCLIPFaiss(
                cache_dir=tmpdir,
                centroid_file="prototypes_k3.npy",
                top_m=2,
            )

            values = metric.get_vector_score(None, meta={"img_path": paths[1]})

            np.testing.assert_allclose(values, np.asarray([0.2, 0.4], dtype=np.float32), atol=1e-6)
            self.assertEqual(values.dtype, np.float32)
            self.assertEqual(values.shape, (2,))
            self.assertAlmostEqual(metric.get_score(None, meta={"img_path": paths[1]}), 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
