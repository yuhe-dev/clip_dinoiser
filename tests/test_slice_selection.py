import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_discovery.selection import (
    SliceSelectionCandidate,
    SliceSelectionThresholds,
    compute_gmm_bic,
    evaluate_gmm_candidate,
    evaluate_vmf_candidate,
    generate_candidate_ks,
    select_best_candidate,
)


class SliceSelectionTests(unittest.TestCase):
    def test_generate_candidate_ks_covers_up_to_sixty_four(self):
        self.assertEqual(generate_candidate_ks(4, 64), [4, 6, 8, 12, 16, 24, 32, 48, 64])

    def test_compute_gmm_bic_penalizes_more_complex_models(self):
        bic_k2 = compute_gmm_bic(log_likelihood=-10.0, n_samples=100, n_features=4, num_slices=2)
        bic_k4 = compute_gmm_bic(log_likelihood=-10.0, n_samples=100, n_features=4, num_slices=4)

        self.assertLess(bic_k2, bic_k4)

    def test_evaluate_gmm_candidate_reports_selection_metrics(self):
        matrix = np.asarray(
            [
                [-2.0, -2.0],
                [-1.8, -2.2],
                [2.0, 2.1],
                [2.2, 1.9],
            ],
            dtype=np.float32,
        )
        thresholds = SliceSelectionThresholds(
            min_slice_weight=0.10,
            min_hard_count=1,
            min_avg_max_membership=0.50,
            max_avg_entropy=1.0,
            min_coherence=0.0,
            bic_relative_tolerance=0.01,
        )

        candidate = evaluate_gmm_candidate(
            matrix=matrix,
            sample_ids=["a", "b", "c", "d"],
            num_slices=2,
            thresholds=thresholds,
            seed=0,
            max_iters=50,
        )

        self.assertEqual(candidate.num_slices, 2)
        self.assertTrue(np.isfinite(candidate.bic))
        self.assertGreater(candidate.avg_max_membership, 0.5)
        self.assertLessEqual(candidate.avg_entropy, 1.0)
        self.assertGreaterEqual(candidate.mean_coherence, candidate.min_coherence)
        self.assertTrue(candidate.admissible)

    def test_evaluate_vmf_candidate_reports_selection_metrics(self):
        matrix = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.98, 0.12, 0.0],
                [0.97, -0.10, 0.02],
                [0.0, 1.0, 0.0],
                [0.05, 0.99, 0.0],
                [-0.08, 0.96, 0.02],
            ],
            dtype=np.float32,
        )
        matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
        thresholds = SliceSelectionThresholds(
            min_slice_weight=0.10,
            min_hard_count=1,
            min_avg_max_membership=0.50,
            max_avg_entropy=1.0,
            min_coherence=0.0,
            bic_relative_tolerance=0.01,
        )

        candidate = evaluate_vmf_candidate(
            matrix=matrix,
            sample_ids=[f"sample_{index}" for index in range(matrix.shape[0])],
            num_slices=2,
            thresholds=thresholds,
            seed=0,
            max_iters=30,
        )

        self.assertEqual(candidate.num_slices, 2)
        self.assertTrue(np.isfinite(candidate.bic))
        self.assertGreater(candidate.avg_max_membership, 0.5)
        self.assertLessEqual(candidate.avg_entropy, 1.0)
        self.assertGreaterEqual(candidate.mean_coherence, candidate.min_coherence)
        self.assertTrue(candidate.admissible)

    def test_select_best_candidate_prefers_lower_bic_with_negative_values_then_smaller_k(self):
        thresholds = SliceSelectionThresholds(
            min_slice_weight=0.10,
            min_hard_count=1,
            min_avg_max_membership=0.50,
            max_avg_entropy=1.0,
            min_coherence=0.50,
            bic_relative_tolerance=0.05,
        )
        candidates = [
            SliceSelectionCandidate(
                num_slices=8,
                bic=-100.0,
                log_likelihood=-30.0,
                min_slice_weight=0.10,
                min_hard_count=10,
                avg_max_membership=0.95,
                avg_entropy=0.05,
                mean_coherence=0.90,
                min_coherence=0.82,
                admissible=True,
                rejection_reasons=[],
            ),
            SliceSelectionCandidate(
                num_slices=16,
                bic=-102.0,
                log_likelihood=-29.0,
                min_slice_weight=0.08,
                min_hard_count=8,
                avg_max_membership=0.96,
                avg_entropy=0.04,
                mean_coherence=0.92,
                min_coherence=0.84,
                admissible=True,
                rejection_reasons=[],
            ),
        ]

        selected = select_best_candidate(candidates, thresholds)

        self.assertEqual(selected.num_slices, 8)

    def test_select_best_candidate_fallback_prefers_fewer_failures_then_higher_coherence(self):
        thresholds = SliceSelectionThresholds(
            min_slice_weight=0.10,
            min_hard_count=1,
            min_avg_max_membership=0.50,
            max_avg_entropy=1.0,
            min_coherence=0.50,
            bic_relative_tolerance=0.05,
        )
        candidates = [
            SliceSelectionCandidate(
                num_slices=12,
                bic=-80.0,
                log_likelihood=-30.0,
                min_slice_weight=0.10,
                min_hard_count=10,
                avg_max_membership=0.95,
                avg_entropy=0.05,
                mean_coherence=0.60,
                min_coherence=0.49,
                admissible=False,
                rejection_reasons=["min_coherence"],
            ),
            SliceSelectionCandidate(
                num_slices=32,
                bic=-100.0,
                log_likelihood=-29.0,
                min_slice_weight=0.10,
                min_hard_count=10,
                avg_max_membership=0.95,
                avg_entropy=0.05,
                mean_coherence=0.58,
                min_coherence=0.47,
                admissible=False,
                rejection_reasons=["min_coherence"],
            ),
        ]

        selected = select_best_candidate(candidates, thresholds)

        self.assertEqual(selected.num_slices, 12)

    def test_select_best_candidate_for_vmf_prefers_smallest_candidate_near_best_plateau(self):
        thresholds = SliceSelectionThresholds(
            min_slice_weight=0.10,
            min_hard_count=1,
            min_avg_max_membership=0.50,
            max_avg_entropy=1.0,
            min_coherence=0.50,
            bic_relative_tolerance=0.05,
        )
        candidates = [
            SliceSelectionCandidate(
                num_slices=16,
                bic=-100.0,
                log_likelihood=172.0,
                min_slice_weight=0.03,
                min_hard_count=1500,
                avg_max_membership=0.99,
                avg_entropy=0.03,
                mean_coherence=0.742,
                min_coherence=0.620,
                admissible=True,
                rejection_reasons=[],
            ),
            SliceSelectionCandidate(
                num_slices=24,
                bic=-110.0,
                log_likelihood=176.0,
                min_slice_weight=0.02,
                min_hard_count=1300,
                avg_max_membership=0.99,
                avg_entropy=0.03,
                mean_coherence=0.764,
                min_coherence=0.672,
                admissible=True,
                rejection_reasons=[],
            ),
            SliceSelectionCandidate(
                num_slices=32,
                bic=-111.0,
                log_likelihood=178.0,
                min_slice_weight=0.02,
                min_hard_count=1200,
                avg_max_membership=0.99,
                avg_entropy=0.03,
                mean_coherence=0.775,
                min_coherence=0.662,
                admissible=True,
                rejection_reasons=[],
            ),
            SliceSelectionCandidate(
                num_slices=48,
                bic=-112.0,
                log_likelihood=182.0,
                min_slice_weight=0.01,
                min_hard_count=600,
                avg_max_membership=0.98,
                avg_entropy=0.03,
                mean_coherence=0.792,
                min_coherence=0.688,
                admissible=True,
                rejection_reasons=[],
            ),
        ]

        selected = select_best_candidate(candidates, thresholds, finder="vmf")

        self.assertEqual(selected.num_slices, 24)


if __name__ == "__main__":
    unittest.main()
