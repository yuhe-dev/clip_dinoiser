import json
import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.feature_utils.data_feature.bundle.io import ProcessedBundleIO, RawBundleIO
from clip_dinoiser.feature_utils.data_feature.bundle.processed_bundle import ProcessedFeatureBundle
from clip_dinoiser.feature_utils.data_feature.bundle.raw_bundle import RawFeatureBundle


class TestFeatureBundleIO(unittest.TestCase):
    def test_raw_bundle_io_uses_existing_quality_filenames(self):
        bundle = RawFeatureBundle(
            dimension_name="quality",
            records=[{"image_rel": "images/train2017/0001.jpg"}],
            stats={"num_samples": 1, "features": {}},
            feature_config={"subset_root": "data/coco_stuff50k"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = RawBundleIO().save(bundle, tmpdir)

            self.assertTrue(paths["records_path"].endswith("quality_raw_features.npy"))
            self.assertTrue(paths["stats_path"].endswith("quality_global_stats.json"))
            self.assertTrue(paths["config_path"].endswith("quality_feature_config.json"))

            loaded = np.load(paths["records_path"], allow_pickle=True)
            self.assertEqual(len(loaded), 1)
            with open(paths["stats_path"], "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f)["num_samples"], 1)

    def test_processed_bundle_io_uses_existing_difficulty_filenames(self):
        bundle = ProcessedFeatureBundle(
            dimension_name="difficulty",
            records=[{"image_rel": "images/train2017/0001.jpg", "features": {}}],
            schema={"schema_version": "difficulty.v1"},
            processing_config={"dimension": "difficulty"},
            summary={"num_samples": 1, "features": {}},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ProcessedBundleIO().save(bundle, tmpdir)

            self.assertTrue(paths["records_path"].endswith("difficulty_processed_features.npy"))
            self.assertTrue(paths["schema_path"].endswith("difficulty_processed_schema.json"))
            self.assertTrue(paths["config_path"].endswith("difficulty_processing_config.json"))
            self.assertTrue(paths["summary_path"].endswith("difficulty_processed_summary.json"))

            loaded = np.load(paths["records_path"], allow_pickle=True)
            self.assertEqual(len(loaded), 1)
            with open(paths["schema_path"], "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f)["schema_version"], "difficulty.v1")


if __name__ == "__main__":
    unittest.main()
