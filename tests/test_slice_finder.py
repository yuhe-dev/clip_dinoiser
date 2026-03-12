import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_discovery.finder import GMMSliceFinder, SoftKMeansSliceFinder
from clip_dinoiser.slice_discovery.types import SliceFindingResult


class SliceFinderTests(unittest.TestCase):
    def test_soft_kmeans_returns_membership_rows_that_sum_to_one(self):
        matrix = np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [5.0, 5.0],
                [5.1, 5.2],
            ],
            dtype=np.float32,
        )
        sample_ids = ["a", "b", "c", "d"]
        finder = SoftKMeansSliceFinder(num_slices=2, seed=0, max_iters=20, temperature=1.0)

        result = finder.fit(matrix, sample_ids=sample_ids)

        self.assertIsInstance(result, SliceFindingResult)
        self.assertEqual(result.membership.shape, (4, 2))
        np.testing.assert_allclose(result.membership.sum(axis=1), np.ones(4, dtype=np.float32), atol=1e-5)
        self.assertEqual(result.hard_assignment.shape, (4,))
        self.assertEqual(result.slice_weights.shape, (2,))
        self.assertEqual(result.sample_ids, sample_ids)

    def test_soft_kmeans_groups_nearby_points_into_same_dominant_slice(self):
        matrix = np.asarray(
            [
                [0.0, 0.0],
                [0.2, 0.1],
                [4.8, 5.0],
                [5.2, 5.1],
            ],
            dtype=np.float32,
        )
        finder = SoftKMeansSliceFinder(num_slices=2, seed=0, max_iters=30, temperature=0.5)

        result = finder.fit(matrix, sample_ids=["a", "b", "c", "d"])

        self.assertEqual(int(result.hard_assignment[0]), int(result.hard_assignment[1]))
        self.assertEqual(int(result.hard_assignment[2]), int(result.hard_assignment[3]))
        self.assertNotEqual(int(result.hard_assignment[0]), int(result.hard_assignment[2]))

    def test_gmm_returns_membership_rows_that_sum_to_one(self):
        matrix = np.asarray(
            [
                [0.0, 0.0],
                [0.1, -0.1],
                [5.0, 5.1],
                [5.2, 4.9],
            ],
            dtype=np.float32,
        )
        sample_ids = ["a", "b", "c", "d"]
        finder = GMMSliceFinder(num_slices=2, seed=0, max_iters=50, covariance_type="diag")

        result = finder.fit(matrix, sample_ids=sample_ids)

        self.assertIsInstance(result, SliceFindingResult)
        self.assertEqual(result.membership.shape, (4, 2))
        np.testing.assert_allclose(result.membership.sum(axis=1), np.ones(4, dtype=np.float32), atol=1e-5)
        self.assertEqual(result.centers.shape, (2, 2))
        self.assertEqual(result.slice_weights.shape, (2,))
        self.assertEqual(result.diagnostics["input_shape"], [4, 2])
        self.assertIn("log_likelihood_trace", result.diagnostics)

    def test_gmm_groups_nearby_points_into_same_dominant_slice(self):
        matrix = np.asarray(
            [
                [-1.0, -1.0],
                [-1.2, -0.8],
                [3.0, 3.1],
                [2.8, 3.2],
            ],
            dtype=np.float32,
        )
        finder = GMMSliceFinder(num_slices=2, seed=0, max_iters=80, covariance_type="diag")

        result = finder.fit(matrix, sample_ids=["a", "b", "c", "d"])

        self.assertEqual(int(result.hard_assignment[0]), int(result.hard_assignment[1]))
        self.assertEqual(int(result.hard_assignment[2]), int(result.hard_assignment[3]))
        self.assertNotEqual(int(result.hard_assignment[0]), int(result.hard_assignment[2]))

    def test_gmm_log_prob_matches_vectorized_reference_formula(self):
        matrix = np.asarray(
            [
                [0.0, 1.0, 2.0],
                [1.5, -0.5, 0.25],
                [3.0, 2.0, -1.0],
            ],
            dtype=np.float32,
        )
        means = np.asarray(
            [
                [0.2, 0.9, 1.8],
                [2.5, 1.0, -0.2],
            ],
            dtype=np.float64,
        )
        covars = np.asarray(
            [
                [1.1, 0.8, 1.3],
                [0.9, 1.4, 0.7],
            ],
            dtype=np.float64,
        )
        finder = GMMSliceFinder(num_slices=2, seed=0, max_iters=5, covariance_type="diag")

        actual = finder._estimate_log_prob(matrix.astype(np.float64), means, covars)

        diff = matrix.astype(np.float64)[:, None, :] - means[None, :, :]
        precision = 1.0 / covars[None, :, :]
        quadratic = (diff * diff * precision).sum(axis=2)
        log_det = np.log(covars).sum(axis=1)
        expected = -0.5 * (matrix.shape[1] * np.log(2.0 * np.pi) + log_det[None, :] + quadratic)

        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)

    def test_soft_kmeans_returns_basic_diagnostics(self):
        matrix = np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [5.0, 5.0],
                [5.1, 5.2],
            ],
            dtype=np.float32,
        )
        finder = SoftKMeansSliceFinder(num_slices=2, seed=0, max_iters=10, temperature=1.0)

        result = finder.fit(matrix, sample_ids=["a", "b", "c", "d"])

        self.assertEqual(result.diagnostics["input_shape"], [4, 2])
        self.assertTrue(result.diagnostics["input_all_finite"])
        self.assertGreaterEqual(result.diagnostics["membership_row_sum_min"], 0.9999)
        self.assertLessEqual(result.diagnostics["membership_row_sum_max"], 1.0001)


if __name__ == "__main__":
    unittest.main()
