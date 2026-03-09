import json
import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.extract_quality_raw_features import (
    compute_global_stats,
    save_quality_feature_bundle,
)
from clip_dinoiser.sanity_check_quality_raw_features import (
    compute_feature_summary,
    compute_quality_bundle_summary,
)


class TestQualityRawFeatureScripts(unittest.TestCase):
    def _make_records(self):
        return [
            {
                "image_rel": "images/train2017/0001.jpg",
                "annotation_rel": "annotations/train2017/0001_labelTrainIds.png",
                "laplacian_raw": np.asarray([1.0, 3.0], dtype=np.float32),
                "noise_pca_raw": np.asarray([0.2, 0.6], dtype=np.float32),
                "bga_raw": np.asarray([0.1, 0.9], dtype=np.float32),
            },
            {
                "image_rel": "images/train2017/0002.jpg",
                "annotation_rel": "annotations/train2017/0002_labelTrainIds.png",
                "laplacian_raw": np.asarray([2.0], dtype=np.float32),
                "noise_pca_raw": np.asarray([0.4], dtype=np.float32),
                "bga_raw": np.asarray([], dtype=np.float32),
            },
        ]

    def test_compute_global_stats_aggregates_lengths_and_ranges(self):
        stats = compute_global_stats(self._make_records())

        self.assertEqual(stats["num_samples"], 2)
        self.assertEqual(stats["features"]["laplacian_raw"]["global_min"], 1.0)
        self.assertEqual(stats["features"]["laplacian_raw"]["global_max"], 3.0)
        self.assertEqual(stats["features"]["laplacian_raw"]["total_values"], 3)
        self.assertEqual(stats["features"]["bga_raw"]["empty_samples"], 1)

    def test_save_quality_feature_bundle_writes_records_stats_and_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_root = os.path.join(tmpdir, "quality_raw")
            records = self._make_records()
            stats = compute_global_stats(records)

            records_path, stats_path, config_path = save_quality_feature_bundle(
                output_root=out_root,
                records=records,
                stats=stats,
                subset_root="data/coco_stuff50k",
                index_path="data/coco_stuff50k/sample_index.npy",
                feature_meta={"patch_size": 32, "stride": 16},
            )

            self.assertTrue(os.path.exists(records_path))
            self.assertTrue(os.path.exists(stats_path))
            self.assertTrue(os.path.exists(config_path))

            loaded_records = np.load(records_path, allow_pickle=True)
            self.assertEqual(len(loaded_records), 2)
            self.assertEqual(loaded_records[0]["image_rel"], "images/train2017/0001.jpg")

            with open(stats_path, "r", encoding="utf-8") as f:
                loaded_stats = json.loads(f.read())
            self.assertEqual(loaded_stats["features"]["noise_pca_raw"]["total_values"], 3)

            with open(config_path, "r", encoding="utf-8") as f:
                loaded_config = json.loads(f.read())
            self.assertEqual(loaded_config["subset_root"], "data/coco_stuff50k")
            self.assertEqual(loaded_config["feature_meta"]["patch_size"], 32)

    def test_compute_feature_summary_reports_length_and_value_stats(self):
        summary = compute_feature_summary(
            [np.asarray([1.0, 3.0], dtype=np.float32), np.asarray([2.0], dtype=np.float32)],
            sample_limit=1,
        )

        self.assertEqual(summary["num_samples"], 2)
        self.assertEqual(summary["length"]["max"], 2)
        self.assertEqual(summary["values"]["min"], 1.0)
        self.assertEqual(len(summary["sample_values"]), 1)

    def test_compute_quality_bundle_summary_summarizes_all_features(self):
        summary = compute_quality_bundle_summary(self._make_records(), sample_limit=1)

        self.assertIn("laplacian_raw", summary["features"])
        self.assertIn("noise_pca_raw", summary["features"])
        self.assertIn("bga_raw", summary["features"])
        self.assertEqual(summary["features"]["bga_raw"]["empty_samples"], 1)


if __name__ == "__main__":
    unittest.main()
