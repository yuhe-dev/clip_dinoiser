import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from clip_dinoiser.sanity_check_laplacian_vector import (
    build_argparser,
    compute_vector_stats,
    parse_sizes,
    pca_2d,
)


class TestLaplacianVectorSanityScript(unittest.TestCase):
    def test_argparse_accepts_no_progress(self):
        parser = build_argparser()
        args = parser.parse_args(["--sizes", "200,2000", "--no-progress"])
        self.assertEqual(args.sizes, "200,2000")
        self.assertTrue(args.no_progress)

    def test_parse_sizes(self):
        self.assertEqual(parse_sizes("200,2000,20000"), [200, 2000, 20000])
        self.assertEqual(parse_sizes(" 5 , 10 "), [5, 10])

    def test_compute_vector_stats_shape_and_l1(self):
        vectors = np.array([
            [0.2, 0.3, 0.5],
            [0.0, 0.5, 0.5],
        ], dtype=np.float32)
        stats = compute_vector_stats(vectors)

        self.assertEqual(stats["num_samples"], 2)
        self.assertEqual(stats["vector_dim"], 3)
        self.assertAlmostEqual(stats["l1_sum_mean"], 1.0, places=6)
        self.assertIn("per_dim_mean", stats)
        self.assertEqual(len(stats["per_dim_mean"]), 3)

    def test_pca_2d_output_shape(self):
        X = np.random.RandomState(0).randn(20, 8).astype(np.float32)
        Y = pca_2d(X)
        self.assertEqual(Y.shape, (20, 2))


if __name__ == "__main__":
    unittest.main()
