import json
import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.extract_difficulty_raw_features import (
    compute_global_stats,
    limit_subset_records,
    load_coco_stuff_classes,
    save_difficulty_feature_bundle,
)
from clip_dinoiser.sanity_check_difficulty_raw_features import (
    compute_difficulty_bundle_summary,
    compute_feature_summary,
)


class TestDifficultyRawFeatureScripts(unittest.TestCase):
    def _make_records(self):
        return [
            {
                "image_rel": "images/train2017/0001.jpg",
                "annotation_rel": "annotations/train2017/0001_labelTrainIds.png",
                "small_ratio_raw": np.asarray([0.1] * 16, dtype=np.float32),
                "visual_semantic_gap_raw": np.asarray([0.2, 0.6], dtype=np.float32),
                "empirical_iou_raw": np.asarray([0.3, 0.8], dtype=np.float32),
            },
            {
                "image_rel": "images/train2017/0002.jpg",
                "annotation_rel": "annotations/train2017/0002_labelTrainIds.png",
                "small_ratio_raw": np.asarray([0.0] * 16, dtype=np.float32),
                "visual_semantic_gap_raw": np.asarray([0.4], dtype=np.float32),
                "empirical_iou_raw": np.asarray([], dtype=np.float32),
            },
        ]

    def test_load_coco_stuff_classes_from_dataset_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = os.path.join(tmpdir, "coco_stuff.py")
            with open(module_path, "w", encoding="utf-8") as f:
                f.write(
                    "class COCOStuffDataset:\n"
                    "    CLASSES = ('cat', 'dog', 'tree')\n"
                )
            classes = load_coco_stuff_classes(module_path)
            self.assertEqual(classes, ["cat", "dog", "tree"])

    def test_compute_global_stats_aggregates_difficulty_features(self):
        stats = compute_global_stats(self._make_records())

        self.assertEqual(stats["num_samples"], 2)
        self.assertEqual(stats["features"]["small_ratio_raw"]["total_values"], 32)
        self.assertAlmostEqual(stats["features"]["visual_semantic_gap_raw"]["global_max"], 0.6, places=6)
        self.assertEqual(stats["features"]["empirical_iou_raw"]["empty_samples"], 1)

    def test_save_difficulty_feature_bundle_writes_records_stats_and_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_root = os.path.join(tmpdir, "difficulty_raw")
            records = self._make_records()
            stats = compute_global_stats(records)

            records_path, stats_path, config_path = save_difficulty_feature_bundle(
                output_root=out_root,
                records=records,
                stats=stats,
                subset_root="data/coco_stuff50k",
                index_path="data/coco_stuff50k/sample_index.npy",
                feature_meta={"model_cfg": "configs/maskclip.yaml"},
                class_names=["cat", "dog", "tree"],
            )

            self.assertTrue(os.path.exists(records_path))
            self.assertTrue(os.path.exists(stats_path))
            self.assertTrue(os.path.exists(config_path))

            loaded_records = np.load(records_path, allow_pickle=True)
            self.assertEqual(len(loaded_records), 2)

            with open(stats_path, "r", encoding="utf-8") as f:
                loaded_stats = json.loads(f.read())
            self.assertEqual(loaded_stats["features"]["empirical_iou_raw"]["empty_samples"], 1)

            with open(config_path, "r", encoding="utf-8") as f:
                loaded_config = json.loads(f.read())
            self.assertEqual(loaded_config["class_names_count"], 3)
            self.assertEqual(loaded_config["feature_meta"]["model_cfg"], "configs/maskclip.yaml")

    def test_compute_feature_summary_reports_length_and_value_stats(self):
        summary = compute_feature_summary(
            [np.asarray([0.2, 0.6], dtype=np.float32), np.asarray([0.4], dtype=np.float32)],
            sample_limit=1,
        )

        self.assertEqual(summary["num_samples"], 2)
        self.assertEqual(summary["length"]["max"], 2)
        self.assertAlmostEqual(summary["values"]["max"], 0.6, places=6)
        self.assertEqual(len(summary["sample_values"]), 1)

    def test_compute_difficulty_bundle_summary_summarizes_all_features(self):
        summary = compute_difficulty_bundle_summary(self._make_records(), sample_limit=1)

        self.assertIn("small_ratio_raw", summary["features"])
        self.assertIn("visual_semantic_gap_raw", summary["features"])
        self.assertIn("empirical_iou_raw", summary["features"])
        self.assertEqual(summary["features"]["empirical_iou_raw"]["empty_samples"], 1)

    def test_limit_subset_records_truncates_only_when_positive(self):
        records = [{"id": idx} for idx in range(5)]

        self.assertEqual(limit_subset_records(records, limit=None), records)
        self.assertEqual(limit_subset_records(records, limit=0), records)
        self.assertEqual(limit_subset_records(records, limit=-1), records)
        self.assertEqual(limit_subset_records(records, limit=2), records[:2])


if __name__ == "__main__":
    unittest.main()
