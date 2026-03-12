import json
import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.visualize_feature_landscape import (
    assemble_feature_landscape_matrix,
    export_landscape_matrix_json,
    pool_profile_sequence,
    resample_sequence,
)


class TestFeatureLandscapeVisualization(unittest.TestCase):
    def test_resample_sequence_expands_short_hist_to_width_16(self):
        seq = np.asarray([0.0, 0.5, 1.0, 0.0], dtype=np.float32)

        out = resample_sequence(seq, target_width=16)

        self.assertEqual(out.shape, (16,))
        self.assertAlmostEqual(float(out[0]), 0.0, places=6)
        self.assertAlmostEqual(float(out[-1]), 0.0, places=6)
        self.assertTrue(np.all(out >= 0.0))

    def test_pool_profile_sequence_reduces_rank_profile_to_width_16(self):
        seq = np.linspace(0.0, 49.0, 50, dtype=np.float32)

        out = pool_profile_sequence(seq, target_width=16)

        self.assertEqual(out.shape, (16,))
        self.assertAlmostEqual(float(out[0]), 1.5, places=6)
        self.assertAlmostEqual(float(out[-1]), 48.0, places=6)
        self.assertTrue(np.all(out[1:] >= out[:-1]))

    def test_assemble_feature_landscape_matrix_builds_8x16_matrix(self):
        quality_record = {
            "image_rel": "images/train2017/0001.jpg",
            "features": {
                "laplacian": {"hist": np.asarray([0.1] * 12, dtype=np.float32)},
                "noise_pca": {"hist": np.asarray([0.2] * 12, dtype=np.float32)},
                "bga": {"hist": np.asarray([0.3] * 8, dtype=np.float32)},
            },
        }
        difficulty_record = {
            "image_rel": "images/train2017/0001.jpg",
            "features": {
                "small_ratio": {"delta_profile": np.asarray([0.05] * 16, dtype=np.float32)},
                "visual_semantic_gap": {"hist": np.asarray([0.15] * 12, dtype=np.float32)},
                "empirical_iou": {"hist": np.asarray([0.25] * 8, dtype=np.float32)},
            },
        }
        coverage_record = {
            "image_rel": "images/train2017/0001.jpg",
            "features": {
                "knn_local_density": {"profile": np.linspace(0.1, 0.5, 50, dtype=np.float32)},
                "prototype_distance": {"profile": np.linspace(0.2, 0.8, 50, dtype=np.float32)},
            },
        }

        matrix, meta = assemble_feature_landscape_matrix(
            quality_record=quality_record,
            difficulty_record=difficulty_record,
            coverage_record=coverage_record,
            target_width=16,
        )

        self.assertEqual(matrix.shape, (8, 16))
        self.assertEqual(
            meta["row_labels"],
            [
                "quality.laplacian",
                "quality.noise_pca",
                "quality.bga",
                "difficulty.small_ratio",
                "difficulty.visual_semantic_gap",
                "difficulty.empirical_iou",
                "coverage.knn_local_density",
                "coverage.prototype_distance",
            ],
        )
        self.assertEqual(meta["group_slices"]["quality"], [0, 3])
        self.assertEqual(meta["group_slices"]["difficulty"], [3, 6])
        self.assertEqual(meta["group_slices"]["coverage"], [6, 8])

    def test_export_landscape_matrix_json_writes_expected_metadata(self):
        matrix = np.ones((8, 16), dtype=np.float32)
        meta = {
            "image_rel": "images/train2017/0001.jpg",
            "row_labels": ["a"] * 8,
            "group_slices": {"quality": [0, 3], "difficulty": [3, 6], "coverage": [6, 8]},
            "target_width": 16,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "matrix.json")
            export_landscape_matrix_json(out_path, matrix, meta)

            with open(out_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.assertEqual(payload["image_rel"], "images/train2017/0001.jpg")
            self.assertEqual(payload["shape"], [8, 16])
            self.assertEqual(payload["target_width"], 16)
            self.assertEqual(len(payload["matrix"]), 8)


if __name__ == "__main__":
    unittest.main()
