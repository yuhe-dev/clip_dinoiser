import math
import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.feature_utils.data_feature.postprocess.encoders import (
    DistributionFeatureEncoder,
    ProfileFeatureEncoder,
)


class TestFeaturePostprocessEncoders(unittest.TestCase):
    def test_distribution_encoder_matches_existing_hist_encoding(self):
        spec = {
            "encoding": "distribution",
            "value_transform": "identity",
            "num_bins": 4,
            "range_mode": "fixed",
            "range_params": {"min": 0.0, "max": 4.0},
            "summary_fields": {
                "q50": "50th percentile of raw values",
                "low_mass": "sum of histogram bins 0-1",
            },
            "model_input_fields": ["hist", "q50"],
        }
        encoder = DistributionFeatureEncoder(spec)
        encoder.fit([np.asarray([0.5, 1.5, 2.5, 3.5], dtype=np.float32)])

        encoded = encoder.transform(np.asarray([0.5, 1.5, 2.5, 3.5], dtype=np.float32), {})

        np.testing.assert_allclose(
            encoded["hist"],
            np.asarray([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
        )
        self.assertAlmostEqual(encoded["summary"]["q50"], 2.0, places=6)
        self.assertAlmostEqual(encoded["summary"]["low_mass"], 0.5, places=6)

    def test_profile_encoder_uses_source_count_key_when_present(self):
        spec = {
            "encoding": "profile",
            "value_transform": "identity",
            "source_count_key": "small_ratio_num_values",
            "summary_fields": {
                "first_active_bin": "index of the first non-zero delta bin normalized to [0, 1]",
                "mass_small_extreme": "sum of delta_profile bins 0-3",
            },
            "model_input_fields": ["delta_profile", "log_num_values"],
        }
        encoder = ProfileFeatureEncoder(spec)

        encoded = encoder.transform(
            np.asarray([0.0, 0.25, 0.75, 1.0], dtype=np.float32),
            {"small_ratio_num_values": 7},
        )

        self.assertEqual(encoded["num_values"], 7)
        self.assertAlmostEqual(encoded["log_num_values"], math.log1p(7), places=6)
        np.testing.assert_allclose(
            encoded["delta_profile"],
            np.asarray([0.0, 0.25, 0.50, 0.25], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
