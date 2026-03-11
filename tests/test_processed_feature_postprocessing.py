import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.postprocess_feature_bundles import (
    encode_distribution_feature,
    encode_profile_feature,
    fit_distribution_bin_edges,
    process_dimension_records,
    save_processed_bundle,
)


class TestProcessedFeaturePostprocessing(unittest.TestCase):
    def test_encode_distribution_feature_normalizes_hist_and_summary(self):
        spec = {
            "encoding": "distribution",
            "value_transform": "identity",
            "num_bins": 4,
            "range_mode": "fixed",
            "range_params": {"min": 0.0, "max": 4.0},
            "summary_fields": {
                "mean": "mean of raw values",
                "std": "standard deviation of raw values",
                "q10": "10th percentile of raw values",
                "q50": "50th percentile of raw values",
                "q90": "90th percentile of raw values",
                "low_mass": "sum of histogram bins 0-1",
                "high_mass": "sum of histogram bins 2-3"
            },
            "model_input_fields": ["hist", "log_num_values", "empty_flag", "q50"]
        }
        raw = np.asarray([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
        edges = fit_distribution_bin_edges([raw], spec)

        encoded = encode_distribution_feature(raw, spec, edges)

        np.testing.assert_allclose(
            encoded["hist"],
            np.asarray([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
        )
        self.assertEqual(encoded["empty_flag"], 0)
        self.assertEqual(encoded["num_values"], 4)
        self.assertAlmostEqual(encoded["log_num_values"], math.log1p(4), places=6)
        self.assertAlmostEqual(encoded["summary"]["q50"], 2.0, places=6)
        self.assertAlmostEqual(encoded["summary"]["low_mass"], 0.5, places=6)
        self.assertEqual(encoded["model_input_fields"], ["hist", "log_num_values", "empty_flag", "q50"])

    def test_encode_profile_feature_produces_delta_profile_and_small_ratio_summary(self):
        spec = {
            "encoding": "profile",
            "value_transform": "identity",
            "derived_profile_field": "delta_profile",
            "summary_fields": {
                "first_active_bin": "index of the first non-zero delta bin normalized to [0, 1]",
                "mass_small_extreme": "sum of delta_profile bins 0-3",
                "mass_small_mid": "sum of delta_profile bins 4-9"
            },
            "model_input_fields": ["delta_profile", "log_num_values", "empty_flag"]
        }
        raw = np.asarray([0.0, 0.0, 0.25, 0.50, 0.80, 1.0], dtype=np.float32)

        encoded = encode_profile_feature(raw, spec, feature_name="small_ratio")

        np.testing.assert_allclose(
            encoded["delta_profile"],
            np.asarray([0.0, 0.0, 0.25, 0.25, 0.30, 0.20], dtype=np.float32),
        )
        self.assertEqual(encoded["num_values"], 6)
        self.assertAlmostEqual(encoded["summary"]["first_active_bin"], 0.4, places=6)
        self.assertAlmostEqual(encoded["summary"]["mass_small_extreme"], 0.5, places=6)
        self.assertAlmostEqual(encoded["summary"]["mass_small_mid"], 0.5, places=6)

    def test_process_dimension_records_and_save_bundle_write_expected_files(self):
        dimension_schema = {
            "schema_version": "difficulty.v1",
            "features": {
                "small_ratio": {
                    "raw_key": "small_ratio_raw",
                    "encoding": "profile",
                    "value_transform": "identity",
                    "derived_profile_field": "delta_profile",
                    "summary_fields": {
                        "first_active_bin": "index of the first non-zero delta bin normalized to [0, 1]",
                        "mass_small_extreme": "sum of delta_profile bins 0-3",
                        "mass_small_mid": "sum of delta_profile bins 4-9"
                    },
                    "model_input_fields": ["delta_profile", "log_num_values", "empty_flag"]
                },
                "empirical_iou": {
                    "raw_key": "empirical_iou_raw",
                    "encoding": "distribution",
                    "value_transform": "identity",
                    "num_bins": 4,
                    "range_mode": "fixed",
                    "range_params": {"min": 0.0, "max": 1.0},
                    "summary_fields": {
                        "q10": "10th percentile of raw values",
                        "q50": "50th percentile of raw values",
                        "low_iou_mass": "sum of histogram bins 0-1",
                        "high_iou_mass": "sum of histogram bins 2-3"
                    },
                    "model_input_fields": ["hist", "q50", "low_iou_mass"]
                }
            }
        }
        records = [
            {
                "image_rel": "images/train2017/0001.jpg",
                "annotation_rel": "annotations/train2017/0001_labelTrainIds.png",
                "small_ratio_raw": np.asarray([0.0, 0.2, 0.6, 1.0], dtype=np.float32),
                "empirical_iou_raw": np.asarray([0.2, 0.8], dtype=np.float32)
            }
        ]

        processed = process_dimension_records(records, dimension_schema)
        self.assertEqual(len(processed), 1)
        self.assertIn("small_ratio", processed[0]["features"])
        self.assertIn("empirical_iou", processed[0]["features"])

        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_processed_bundle(
                output_root=tmpdir,
                dimension_name="difficulty",
                processed_records=processed,
                dimension_schema=dimension_schema,
                source_records_path="difficulty_raw_features.npy",
                source_stats_path="difficulty_global_stats.json",
                source_config_path="difficulty_feature_config.json",
                schema_source_path="docs/feature_schema/unified_processed_feature_schema.json",
            )

            self.assertTrue(os.path.exists(result["records_path"]))
            self.assertTrue(os.path.exists(result["schema_path"]))
            self.assertTrue(os.path.exists(result["config_path"]))
            self.assertTrue(os.path.exists(result["summary_path"]))

            loaded_records = np.load(result["records_path"], allow_pickle=True)
            self.assertEqual(len(loaded_records), 1)
            self.assertEqual(loaded_records[0]["schema_version"], "difficulty.v1")

            with open(result["config_path"], "r", encoding="utf-8") as f:
                config = json.load(f)
            self.assertEqual(config["dimension"], "difficulty")
            self.assertEqual(config["source_records_path"], "difficulty_raw_features.npy")


if __name__ == "__main__":
    unittest.main()
