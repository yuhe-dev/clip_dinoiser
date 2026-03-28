import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_target_search_review import _shift_quality_laplacian_target, build_parser
from clip_dinoiser.slice_remix.prior_graph import TargetPortraitSpec


class RunTargetSearchReviewTests(unittest.TestCase):
    def test_parser_defaults_to_raw_pool_target_mode(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parser.parse_args(["--output-dir", tmpdir])
        self.assertEqual(args.target_mode, "raw_pool")

    def test_shift_quality_laplacian_target_preserves_other_targets(self):
        original = TargetPortraitSpec(
            shape_targets={
                "quality.laplacian": np.asarray([0.7, 0.2, 0.1], dtype=np.float32),
                "coverage.knn_local_density": np.asarray([0.4, 0.6], dtype=np.float32),
            },
            scalar_targets={
                "quality.laplacian": np.asarray([0.2, 0.0], dtype=np.float32),
                "coverage.knn_local_density": np.asarray([0.3], dtype=np.float32),
            },
            block_weights={
                "quality.laplacian": 0.5,
                "coverage.knn_local_density": 0.5,
            },
            source="pool_initialized",
        )

        shifted = _shift_quality_laplacian_target(original, mass=0.08)

        self.assertEqual(set(shifted.shape_targets.keys()), set(original.shape_targets.keys()))
        self.assertEqual(set(shifted.scalar_targets.keys()), set(original.scalar_targets.keys()))
        self.assertFalse(
            np.allclose(
                shifted.shape_targets["quality.laplacian"],
                original.shape_targets["quality.laplacian"],
            )
        )
        self.assertTrue(
            np.allclose(
                shifted.shape_targets["coverage.knn_local_density"],
                original.shape_targets["coverage.knn_local_density"],
            )
        )
        self.assertTrue(
            np.allclose(
                shifted.scalar_targets["quality.laplacian"],
                original.scalar_targets["quality.laplacian"],
            )
        )
