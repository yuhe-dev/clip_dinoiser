import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_remix.actions import generate_pairwise_candidates, select_pairwise_directions


class SliceRemixActionsTests(unittest.TestCase):
    def test_generate_pairwise_candidates_preserves_simplex(self):
        baseline = np.asarray([0.4, 0.35, 0.25], dtype=np.float32)

        candidates = generate_pairwise_candidates(
            baseline,
            amplitudes=[0.05],
            ordered_pairs=[(0, 1), (2, 1)],
        )

        self.assertEqual(len(candidates), 2)
        for candidate in candidates:
            self.assertAlmostEqual(sum(candidate.target_mixture), 1.0, places=6)
            self.assertTrue(np.all(np.asarray(candidate.target_mixture) >= 0.0))

    def test_generate_pairwise_candidates_records_action_metadata(self):
        baseline = np.asarray([0.4, 0.35, 0.25], dtype=np.float32)

        candidate = generate_pairwise_candidates(
            baseline,
            amplitudes=[0.05],
            ordered_pairs=[(0, 1)],
        )[0]

        self.assertEqual(candidate.receivers, [0])
        self.assertEqual(candidate.donors, [1])
        self.assertAlmostEqual(candidate.amplitude, 0.05, places=6)
        self.assertEqual(candidate.support_size, 2)
        np.testing.assert_allclose(candidate.delta_q, [0.05, -0.05, 0.0])

    def test_select_pairwise_directions_prefers_diverse_portrait_shifts(self):
        baseline = np.asarray([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        portraits = {
            "feature_a": np.asarray([[10.0], [0.0], [0.0], [9.0]], dtype=np.float32),
            "feature_b": np.asarray([[0.0], [0.0], [8.0], [0.0]], dtype=np.float32),
        }

        selected = select_pairwise_directions(
            baseline_mixture=baseline,
            portraits=portraits,
            max_pairs=2,
            ordered_pairs=[(0, 1), (3, 1), (2, 1)],
            min_amplitude=0.05,
        )

        self.assertEqual(selected[0], (0, 1))
        self.assertEqual(selected[1], (2, 1))


if __name__ == "__main__":
    unittest.main()
