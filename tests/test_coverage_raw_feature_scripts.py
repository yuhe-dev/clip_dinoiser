import json
import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.build_coverage_prototypes import save_prototype_bundle
from clip_dinoiser.extract_coverage_embeddings import (
    build_embedding_meta,
    limit_subset_records,
    save_embedding_assets,
)
from clip_dinoiser.extract_coverage_raw_features import compute_coverage_global_stats
from clip_dinoiser.sanity_check_coverage_raw_features import compute_coverage_bundle_summary


class TestCoverageRawFeatureScripts(unittest.TestCase):
    def _make_records(self):
        return [
            {
                "image_rel": "images/train2017/0001.jpg",
                "annotation_rel": "annotations/train2017/0001_labelTrainIds.png",
                "knn_neighbor_distances_raw": np.asarray([0.1, 0.2, 0.4], dtype=np.float32),
                "prototype_distances_raw": np.asarray([0.3, 0.6], dtype=np.float32),
            },
            {
                "image_rel": "images/train2017/0002.jpg",
                "annotation_rel": "annotations/train2017/0002_labelTrainIds.png",
                "knn_neighbor_distances_raw": np.asarray([0.5], dtype=np.float32),
                "prototype_distances_raw": np.asarray([], dtype=np.float32),
            },
        ]

    def test_embedding_metadata_writer_saves_paths_and_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = os.path.join(tmpdir, "visual_embedding")
            embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            clip_paths = [
                "/dataset/images/train2017/0001.jpg",
                "/dataset/images/train2017/0002.jpg",
            ]

            emb_path, paths_path, config_path = save_embedding_assets(
                output_root=output_root,
                embeddings=embeddings,
                clip_paths=clip_paths,
                subset_root="data/coco_stuff50k",
                index_path="data/coco_stuff50k/sample_index.npy",
                embedding_meta={"clip_model": "ViT-B-16", "clip_pretrained": "laion2b_s34b_b88k"},
            )

            self.assertTrue(os.path.exists(emb_path))
            self.assertTrue(os.path.exists(paths_path))
            self.assertTrue(os.path.exists(config_path))

            np.testing.assert_allclose(np.load(emb_path), embeddings)
            with open(paths_path, "r", encoding="utf-8") as f:
                loaded_paths = json.loads(f.read())
            self.assertEqual(loaded_paths, clip_paths)
            with open(config_path, "r", encoding="utf-8") as f:
                loaded_config = json.loads(f.read())
            self.assertEqual(loaded_config["subset_root"], "data/coco_stuff50k")
            self.assertEqual(loaded_config["embedding_meta"]["clip_model"], "ViT-B-16")

    def test_prototype_writer_saves_centroids_and_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = os.path.join(tmpdir, "visual_embedding")
            centroids = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

            centroid_path, config_path = save_prototype_bundle(
                output_root=output_root,
                centroids=centroids,
                prototype_meta={"k": 2, "niter": 5},
                embeddings_file="visual_emb.npy",
                paths_file="clip_paths_abs.json",
            )

            self.assertTrue(os.path.exists(centroid_path))
            self.assertTrue(os.path.exists(config_path))
            np.testing.assert_allclose(np.load(centroid_path), centroids)

            with open(config_path, "r", encoding="utf-8") as f:
                loaded_config = json.loads(f.read())
            self.assertEqual(loaded_config["prototype_meta"]["k"], 2)
            self.assertEqual(loaded_config["embeddings_file"], "visual_emb.npy")

    def test_coverage_summary_reports_lengths_and_ranges(self):
        summary = compute_coverage_bundle_summary(self._make_records(), sample_limit=1)

        self.assertEqual(summary["num_samples"], 2)
        self.assertIn("knn_neighbor_distances_raw", summary["features"])
        self.assertIn("prototype_distances_raw", summary["features"])
        self.assertEqual(summary["features"]["knn_neighbor_distances_raw"]["length"]["max"], 3)
        self.assertAlmostEqual(summary["features"]["prototype_distances_raw"]["values"]["max"], 0.6, places=6)

    def test_limit_subset_records_truncates_embedding_input(self):
        records = [{"image_rel": f"images/train2017/{idx:04d}.jpg"} for idx in range(5)]

        self.assertEqual(limit_subset_records(records, limit=None), records)
        self.assertEqual(limit_subset_records(records, limit=0), records)
        self.assertEqual(limit_subset_records(records, limit=2), records[:2])

        meta = build_embedding_meta(
            clip_model="ViT-B-16",
            clip_pretrained="laion2b_s34b_b88k",
            device="cpu",
            batch_size=8,
        )
        self.assertEqual(meta["clip_model"], "ViT-B-16")
        self.assertEqual(meta["batch_size"], 8)

    def test_compute_coverage_global_stats_aggregates_knn_and_prototype_values(self):
        stats = compute_coverage_global_stats(self._make_records())

        self.assertEqual(stats["num_samples"], 2)
        self.assertEqual(stats["features"]["knn_neighbor_distances_raw"]["total_values"], 4)
        self.assertAlmostEqual(
            stats["features"]["knn_neighbor_distances_raw"]["global_max"],
            0.5,
            places=6,
        )
        self.assertEqual(stats["features"]["prototype_distances_raw"]["empty_samples"], 1)


if __name__ == "__main__":
    unittest.main()
