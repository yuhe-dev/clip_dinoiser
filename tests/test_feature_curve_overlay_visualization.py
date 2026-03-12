import json
import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.visualize_feature_curve_overlay import (
    build_curve_style_metadata,
    export_curve_overlay_json,
    build_ridge_layout,
)


class TestFeatureCurveOverlayVisualization(unittest.TestCase):
    def test_build_curve_style_metadata_returns_grouped_colors_and_labels(self):
        row_labels = [
            "quality.laplacian",
            "quality.noise_pca",
            "quality.bga",
            "difficulty.small_ratio",
            "difficulty.visual_semantic_gap",
            "difficulty.empirical_iou",
            "coverage.knn_local_density",
            "coverage.prototype_distance",
        ]

        meta = build_curve_style_metadata(row_labels)

        self.assertEqual(len(meta["curve_labels"]), 8)
        self.assertEqual(meta["curve_labels"][0], "Laplacian")
        self.assertEqual(meta["curve_labels"][3], "Small Ratio")
        self.assertEqual(set(meta["legend_labels"].keys()), {"quality", "difficulty", "coverage"})
        self.assertEqual(len(meta["curve_colors"]), 8)
        self.assertNotEqual(meta["curve_colors"][0], meta["curve_colors"][3])

    def test_export_curve_overlay_json_writes_matrix_and_style_metadata(self):
        matrix = np.ones((8, 16), dtype=np.float32)
        meta = {
            "image_rel": "images/train2017/0001.jpg",
            "row_labels": ["quality.laplacian"] * 8,
            "target_width": 16,
        }
        style_meta = {
            "curve_labels": ["Laplacian"] * 8,
            "curve_colors": ["#111111"] * 8,
            "legend_labels": {"quality": "Quality", "difficulty": "Difficulty", "coverage": "Coverage"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "curve_overlay.json")
            export_curve_overlay_json(out_path, matrix, meta, style_meta)

            with open(out_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.assertEqual(payload["image_rel"], "images/train2017/0001.jpg")
            self.assertEqual(payload["shape"], [8, 16])
            self.assertEqual(len(payload["curve_labels"]), 8)
            self.assertEqual(payload["legend_labels"]["quality"], "Quality")

    def test_build_ridge_layout_offsets_curves_by_group_with_larger_group_gaps(self):
        matrix = np.asarray([[0.1] * 16] * 8, dtype=np.float32)
        row_labels = [
            "quality.laplacian",
            "quality.noise_pca",
            "quality.bga",
            "difficulty.small_ratio",
            "difficulty.visual_semantic_gap",
            "difficulty.empirical_iou",
            "coverage.knn_local_density",
            "coverage.prototype_distance",
        ]

        layout = build_ridge_layout(matrix, row_labels)

        self.assertEqual(layout["baselines"].shape, (8,))
        self.assertTrue(layout["baselines"][3] - layout["baselines"][2] > layout["baselines"][1] - layout["baselines"][0])
        self.assertTrue(layout["baselines"][6] - layout["baselines"][5] > layout["baselines"][4] - layout["baselines"][3])
        self.assertEqual(layout["group_centers"]["quality"] < layout["group_centers"]["difficulty"] < layout["group_centers"]["coverage"], True)


if __name__ == "__main__":
    unittest.main()
