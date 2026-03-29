import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_target_search_review import (
    _runtime_summary_lines,
    _shift_quality_laplacian_target,
    build_parser,
)
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

    def test_runtime_summary_lines_cover_graph_layers_and_progress_paths(self):
        payload = SimpleNamespace(
            nodes=[object(), object(), object()],
            edges=[
                SimpleNamespace(admissible=True),
                SimpleNamespace(admissible=False),
                SimpleNamespace(admissible=True),
            ],
        )
        trace = {
            "root_id": "n0",
            "nodes": [
                {
                    "node_id": "n0",
                    "parent_id": None,
                    "depth": 0,
                    "progress": 0.0,
                    "node_type": "root",
                },
                {
                    "node_id": "n1",
                    "parent_id": "n0",
                    "depth": 1,
                    "progress": 0.21,
                    "node_type": "partial",
                },
                {
                    "node_id": "n2",
                    "parent_id": "n1",
                    "depth": 2,
                    "progress": 0.34,
                    "node_type": "completed",
                },
                {
                    "node_id": "n3",
                    "parent_id": "n0",
                    "depth": 1,
                    "progress": 0.18,
                    "node_type": "completed",
                },
            ],
            "layer_summaries": [
                {
                    "depth": 0,
                    "beam_in": 1,
                    "expanded_children": 2,
                    "deduped_children": 2,
                    "beam_out": 2,
                    "best_parent_progress": 0.0,
                    "best_child_progress": 0.21,
                    "pruned_summary": {"proposal_pruned": 4},
                    "stopped": None,
                }
            ],
        }

        lines = _runtime_summary_lines(
            payload=payload,
            trace=trace,
            baseline_gap=12.5,
            candidate_count=2,
        )
        joined = "\n".join(lines)

        self.assertIn("prior graph: nodes=3 admissible_edges=2", joined)
        self.assertIn("depth 0: beam_in=1 expanded=2 deduped=2 beam_out=2", joined)
        self.assertIn("leaf nodes: total=2 completed=2 max_depth=2", joined)
        self.assertIn("n0(0.0000) -> n1(0.2100) -> n2(0.3400)", joined)
