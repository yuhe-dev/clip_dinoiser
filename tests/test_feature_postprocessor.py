import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.feature_utils.data_feature.bundle.raw_bundle import RawFeatureBundle
from clip_dinoiser.feature_utils.data_feature.postprocess.processor import FeaturePostprocessor


class TestFeaturePostprocessor(unittest.TestCase):
    def test_feature_postprocessor_returns_processed_bundle_with_existing_record_shape(self):
        raw_bundle = RawFeatureBundle(
            dimension_name="difficulty",
            records=[
                {
                    "image_rel": "images/train2017/0001.jpg",
                    "annotation_rel": "annotations/train2017/0001_labelTrainIds.png",
                    "small_ratio_raw": np.asarray([0.0, 0.2, 0.6, 1.0], dtype=np.float32),
                    "small_ratio_num_values": 3,
                    "empirical_iou_raw": np.asarray([0.2, 0.8], dtype=np.float32),
                }
            ],
            stats={"num_samples": 1, "features": {}},
            feature_config={"subset_root": "data/coco_stuff50k"},
        )
        dimension_schema = {
            "schema_version": "difficulty.v1",
            "features": {
                "small_ratio": {
                    "raw_key": "small_ratio_raw",
                    "source_count_key": "small_ratio_num_values",
                    "encoding": "profile",
                    "value_transform": "identity",
                    "summary_fields": {
                        "first_active_bin": "index of the first non-zero delta bin normalized to [0, 1]",
                        "mass_small_extreme": "sum of delta_profile bins 0-3",
                    },
                    "model_input_fields": ["delta_profile", "log_num_values"],
                },
                "empirical_iou": {
                    "raw_key": "empirical_iou_raw",
                    "encoding": "distribution",
                    "value_transform": "identity",
                    "num_bins": 4,
                    "range_mode": "fixed",
                    "range_params": {"min": 0.0, "max": 1.0},
                    "summary_fields": {
                        "q50": "50th percentile of raw values",
                    },
                    "model_input_fields": ["hist", "q50"],
                },
            },
        }

        bundle = FeaturePostprocessor().process_bundle(raw_bundle, dimension_schema, progress_interval=1, log_fn=lambda msg: None)

        self.assertEqual(bundle.dimension_name, "difficulty")
        self.assertEqual(bundle.schema["schema_version"], "difficulty.v1")
        self.assertIn("small_ratio", bundle.records[0]["features"])
        self.assertIn("empirical_iou", bundle.records[0]["features"])
        self.assertEqual(bundle.records[0]["features"]["small_ratio"]["num_values"], 3)
        self.assertEqual(bundle.summary["num_samples"], 1)


if __name__ == "__main__":
    unittest.main()
