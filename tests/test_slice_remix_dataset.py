import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_remix.dataset import build_response_row


class SliceRemixDatasetTests(unittest.TestCase):
    def test_build_response_row_contains_core_fields(self):
        row = build_response_row(
            baseline_trial_id="trial_0",
            candidate_id="cand_1",
            baseline_mixture=[0.5, 0.5],
            target_mixture=[0.6, 0.4],
            delta_q=[0.1, -0.1],
            delta_phi={"feature_a": [0.2]},
            context={"budget": 1000},
            measured_gain=0.3,
        )

        self.assertEqual(row["baseline_trial_id"], "trial_0")
        self.assertEqual(row["context"]["budget"], 1000)
        self.assertEqual(row["delta_phi"]["feature_a"], [0.2])


if __name__ == "__main__":
    unittest.main()
