import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_remix.policy import (
    compute_importance_weights,
    materialize_budgeted_subset,
    sample_budgeted_subset,
)


class SliceRemixPolicyTests(unittest.TestCase):
    def test_compute_importance_weights_returns_normalized_weights(self):
        memberships = np.asarray(
            [
                [1.0, 0.0],
                [0.2, 0.8],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        target = np.asarray([0.6, 0.4], dtype=np.float32)

        weights = compute_importance_weights(memberships, target)

        self.assertEqual(weights.shape, (3,))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertGreater(weights[0], weights[2])

    def test_sample_budgeted_subset_returns_unique_ids_with_fixed_budget(self):
        sample_ids = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
        weights = np.asarray([0.6, 0.2, 0.1, 0.1], dtype=np.float32)

        selected = sample_budgeted_subset(sample_ids, weights, budget=2, seed=0)

        self.assertEqual(len(selected), 2)
        self.assertEqual(len(set(selected)), 2)

    def test_sample_budgeted_subset_with_target_mixture_tracks_target(self):
        sample_ids = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg"]
        memberships = np.asarray(
            [
                [0.95, 0.05],
                [0.90, 0.10],
                [0.80, 0.20],
                [0.20, 0.80],
                [0.10, 0.90],
                [0.05, 0.95],
            ],
            dtype=np.float32,
        )
        target = np.asarray([0.5, 0.5], dtype=np.float32)
        weights = compute_importance_weights(memberships, target)

        selected = sample_budgeted_subset(
            sample_ids,
            weights,
            budget=2,
            seed=0,
            memberships=memberships,
            target_mixture=target,
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(len(set(selected)), 2)
        selected_idx = [sample_ids.index(sample_id) for sample_id in selected]
        realized = memberships[selected_idx].mean(axis=0)
        self.assertLess(float(np.abs(realized - target).sum()), 0.3)

    def test_materialize_budgeted_subset_coverage_repair_improves_focus_class_coverage(self):
        sample_ids = [f"{name}.jpg" for name in "abcdef"]
        memberships = np.asarray(
            [
                [0.95, 0.05],
                [0.90, 0.10],
                [0.85, 0.15],
                [0.45, 0.55],
                [0.35, 0.65],
                [0.30, 0.70],
            ],
            dtype=np.float32,
        )
        target = np.asarray([0.5, 0.5], dtype=np.float32)
        weights = compute_importance_weights(memberships, target)
        class_presence = np.asarray(
            [
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        )

        baseline = materialize_budgeted_subset(
            sample_ids,
            weights,
            budget=2,
            seed=0,
            memberships=memberships,
            target_mixture=target,
        )
        repaired = materialize_budgeted_subset(
            sample_ids,
            weights,
            budget=2,
            seed=0,
            memberships=memberships,
            target_mixture=target,
            class_presence=class_presence,
            focus_class_indices=[1, 2],
            focus_class_targets=np.asarray([1, 1], dtype=np.int64),
            focus_class_weights=np.asarray([2.0, 2.0], dtype=np.float32),
            coverage_alpha=0.1,
            coverage_repair_budget=8,
        )

        self.assertEqual(len(repaired.selected_ids), 2)
        self.assertEqual(len(baseline.selected_ids), 2)
        self.assertGreaterEqual(repaired.accepted_coverage_swaps, 1)
        self.assertGreater(
            int(np.count_nonzero(repaired.focus_coverage_after)),
            int(np.count_nonzero(repaired.focus_coverage_before)),
        )
        self.assertLessEqual(
            repaired.mixture_l1_after_coverage_repair,
            repaired.mixture_l1_before_coverage_repair + 0.35,
        )


if __name__ == "__main__":
    unittest.main()
