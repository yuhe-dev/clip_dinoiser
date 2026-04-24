import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.research_harness.feature_intervention import _materialize_pair_for_axis, _score_formula_values


class ResearchFeatureInterventionTests(unittest.TestCase):
    def test_score_formula_combines_z_terms(self):
        records = [
            {"features": {"laplacian": {"summary": {"q50": 1.0, "low_sharpness_mass": 0.9}}}},
            {"features": {"laplacian": {"summary": {"q50": 2.0, "low_sharpness_mass": 0.5}}}},
            {"features": {"laplacian": {"summary": {"q50": 3.0, "low_sharpness_mass": 0.1}}}},
        ]
        scores = _score_formula_values(records, "z(laplacian.summary.q50) - z(laplacian.summary.low_sharpness_mass)")
        self.assertEqual(scores.shape[0], 3)
        self.assertGreater(float(scores[-1]), float(scores[0]))

    def test_materialize_pair_produces_high_low_manifests_and_positive_delta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_ids = [f"images/train2017/{index:012d}.jpg" for index in range(10)]
            sample_paths = [os.path.join(tmpdir, f"{index:012d}.jpg") for index in range(10)]
            axis_scores = {
                "quality_sharpness": np.linspace(-2.0, 2.0, 10, dtype=np.float32),
                "difficulty_small_object": np.zeros(10, dtype=np.float32),
                "coverage_density": np.zeros(10, dtype=np.float32),
            }
            class_presence = np.zeros((10, 3), dtype=np.uint8)
            class_presence[:5, 0] = 1
            class_presence[5:, 1] = 1
            pair = _materialize_pair_for_axis(
                axis_id="quality_sharpness",
                pair_seed=0,
                sample_ids=sample_ids,
                sample_paths=sample_paths,
                axis_scores=axis_scores,
                target_axis_id="quality_sharpness",
                candidate_budget=8,
                subset_budget=3,
                anchor_indices=np.asarray([3, 4, 5], dtype=np.int64),
                class_presence_matrix=class_presence,
                output_dir=tmpdir,
                control_family="real_feature_guided",
            )
            self.assertGreater(float(pair["realized_target_delta"]), 0.0)
            self.assertTrue(os.path.exists(pair["high_manifest_path"]))
            self.assertTrue(os.path.exists(pair["low_manifest_path"]))


if __name__ == "__main__":
    unittest.main()
