import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.feature_utils.data_feature.bundle.raw_bundle import RawFeatureBundle
from clip_dinoiser.feature_utils.data_feature.bundle.processed_bundle import ProcessedFeatureBundle
from clip_dinoiser.feature_utils.data_feature.bundle.stats import (
    build_processed_feature_summary,
    build_raw_feature_stats,
)


class TestFeatureBundleModels(unittest.TestCase):
    def test_build_raw_feature_stats_matches_existing_global_stats_shape(self):
        records = [
            {
                "laplacian_raw": np.asarray([1.0, 2.0], dtype=np.float32),
                "noise_pca_raw": np.asarray([], dtype=np.float32),
            },
            {
                "laplacian_raw": np.asarray([], dtype=np.float32),
                "noise_pca_raw": np.asarray([3.0], dtype=np.float32),
            },
        ]

        stats = build_raw_feature_stats(
            records=records,
            feature_keys=("laplacian_raw", "noise_pca_raw"),
        )

        self.assertEqual(stats["num_samples"], 2)
        self.assertEqual(stats["features"]["laplacian_raw"]["total_values"], 2)
        self.assertEqual(stats["features"]["laplacian_raw"]["empty_samples"], 1)
        self.assertEqual(stats["features"]["noise_pca_raw"]["total_values"], 1)
        self.assertEqual(stats["features"]["noise_pca_raw"]["empty_samples"], 1)

    def test_build_processed_feature_summary_matches_existing_processed_summary_shape(self):
        processed_records = [
            {
                "features": {
                    "laplacian": {
                        "empty_flag": 0,
                        "num_values": 3,
                    }
                }
            },
            {
                "features": {
                    "laplacian": {
                        "empty_flag": 1,
                        "num_values": 0,
                    }
                }
            },
        ]

        summary = build_processed_feature_summary(processed_records)

        self.assertEqual(summary["num_samples"], 2)
        self.assertEqual(summary["features"]["laplacian"]["empty_samples"], 1)
        self.assertEqual(summary["features"]["laplacian"]["num_values_min"], 0)
        self.assertEqual(summary["features"]["laplacian"]["num_values_max"], 3)

    def test_raw_and_processed_bundle_store_expected_attributes(self):
        raw_bundle = RawFeatureBundle(
            dimension_name="quality",
            records=[{"image_rel": "images/train2017/0001.jpg"}],
            stats={"num_samples": 1, "features": {}},
            feature_config={"subset_root": "data/coco_stuff50k"},
        )
        processed_bundle = ProcessedFeatureBundle(
            dimension_name="quality",
            records=[{"image_rel": "images/train2017/0001.jpg", "features": {}}],
            schema={"schema_version": "quality.v1"},
            processing_config={"dimension": "quality"},
            summary={"num_samples": 1, "features": {}},
        )

        self.assertEqual(raw_bundle.dimension_name, "quality")
        self.assertEqual(raw_bundle.stats["num_samples"], 1)
        self.assertEqual(processed_bundle.dimension_name, "quality")
        self.assertEqual(processed_bundle.schema["schema_version"], "quality.v1")


if __name__ == "__main__":
    unittest.main()
