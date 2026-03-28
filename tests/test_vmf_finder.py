import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_discovery.finder import VMFSliceFinder


class VMFSliceFinderTests(unittest.TestCase):
    def test_vmf_finder_returns_soft_memberships_and_kappa_diagnostics(self):
        cluster_a = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.98, 0.12, 0.0],
                [0.97, -0.10, 0.02],
            ],
            dtype=np.float32,
        )
        cluster_b = np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.05, 0.99, 0.0],
                [-0.08, 0.96, 0.02],
            ],
            dtype=np.float32,
        )
        matrix = np.concatenate([cluster_a, cluster_b], axis=0)
        matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
        sample_ids = [f"sample_{index}" for index in range(matrix.shape[0])]

        result = VMFSliceFinder(num_slices=2, seed=0, max_iters=30).fit(matrix, sample_ids)

        self.assertEqual(result.membership.shape, (6, 2))
        self.assertEqual(result.centers.shape, (2, 3))
        self.assertTrue(np.allclose(result.membership.sum(axis=1), 1.0, atol=1e-5))
        self.assertTrue(np.isfinite(result.membership).all())
        self.assertTrue(np.isfinite(result.centers).all())
        self.assertIn("log_likelihood_trace", result.diagnostics)
        self.assertIn("mean_kappa_trace", result.diagnostics)
        self.assertGreater(len(result.diagnostics["log_likelihood_trace"]), 0)
        self.assertGreater(len(result.diagnostics["mean_kappa_trace"]), 0)


if __name__ == "__main__":
    unittest.main()
