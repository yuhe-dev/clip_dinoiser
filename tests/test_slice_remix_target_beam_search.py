import os
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_remix.beam_search import (
    SearchEdge,
    TargetBeamSearchConfig,
    generate_target_beam_candidates,
    get_depth_adaptive_amplitudes,
)
from clip_dinoiser.slice_remix import beam_search
from clip_dinoiser.slice_remix.prior_graph import build_pool_target_portrait_spec, build_target_residual_context


class SliceRemixTargetBeamSearchTests(unittest.TestCase):
    def test_get_depth_adaptive_amplitudes_refines_with_depth(self):
        shallow = get_depth_adaptive_amplitudes(
            delta_max=0.12,
            min_transfer_mass=0.03,
            best_amplitude=0.09,
            depth=0,
        )
        deep = get_depth_adaptive_amplitudes(
            delta_max=0.12,
            min_transfer_mass=0.03,
            best_amplitude=0.09,
            depth=3,
        )

        self.assertLess(len(shallow), len(deep))
        self.assertIn(0.09, [round(value, 2) for value in deep])

    def test_generate_target_beam_candidates_prefers_target_improving_edge(self):
        memberships = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        feature_groups = {
            "quality.laplacian": np.asarray(
                [
                    [0.9, 0.1, 3.0, 0.0],
                    [0.8, 0.2, 2.7, 0.0],
                    [0.2, 0.8, 0.2, 0.0],
                    [0.6, 0.4, 1.8, 0.0],
                ],
                dtype=np.float32,
            ),
        }
        feature_label_map = {
            "quality.laplacian": ["hist[0]", "hist[1]", "log_num_values", "empty_flag"],
        }
        pool_target = build_pool_target_portrait_spec(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
        )
        target_context = build_target_residual_context(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
            baseline_sample_indices=[0, 1],
            target_spec=pool_target,
        )
        baseline = target_context.baseline_mixture
        edges = [
            SearchEdge(donor=0, receiver=1, score=0.9, balance_score=0.0, risk_score=0.1, amplitude_band=(0.0, 0.015), fit_score=0.9, bias_score=0.0),
            SearchEdge(donor=0, receiver=2, score=0.2, balance_score=0.0, risk_score=0.1, amplitude_band=(0.0, 0.015), fit_score=0.2, bias_score=0.0),
        ]

        candidates = generate_target_beam_candidates(
            baseline_mixture=baseline,
            edges=edges,
            target_context=target_context,
            config=TargetBeamSearchConfig(max_depth=1, beam_width=4, proposal_edges_per_node=2, lambda_opportunity=0.0),
        )

        self.assertTrue(candidates)
        plan = candidates[0].metadata["plan"]
        self.assertEqual((plan[0]["donor"], plan[0]["receiver"]), (0, 1))
        self.assertLess(float(plan[0]["amplitude"]), 0.03)

    def test_generate_target_beam_candidates_avoids_global_opportunity_rescan_per_child(self):
        memberships = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        feature_groups = {
            "quality.laplacian": np.asarray(
                [
                    [0.9, 0.1, 3.0, 0.0],
                    [0.8, 0.2, 2.7, 0.0],
                    [0.2, 0.8, 0.2, 0.0],
                    [0.6, 0.4, 1.8, 0.0],
                ],
                dtype=np.float32,
            ),
        }
        feature_label_map = {
            "quality.laplacian": ["hist[0]", "hist[1]", "log_num_values", "empty_flag"],
        }
        pool_target = build_pool_target_portrait_spec(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
        )
        target_context = build_target_residual_context(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
            baseline_sample_indices=[0, 1],
            target_spec=pool_target,
        )
        baseline = target_context.baseline_mixture
        edges = [
            SearchEdge(donor=0, receiver=1, score=0.9, balance_score=0.0, risk_score=0.1, amplitude_band=(0.0, 0.015), fit_score=0.9, bias_score=0.0),
            SearchEdge(donor=0, receiver=2, score=0.2, balance_score=0.0, risk_score=0.1, amplitude_band=(0.0, 0.015), fit_score=0.2, bias_score=0.0),
        ]

        with mock.patch.object(beam_search, "_target_opportunity", wraps=beam_search._target_opportunity) as target_opportunity:
            candidates = generate_target_beam_candidates(
                baseline_mixture=baseline,
                edges=edges,
                target_context=target_context,
                config=TargetBeamSearchConfig(max_depth=1, beam_width=4, proposal_edges_per_node=2),
            )

        self.assertTrue(candidates)
        self.assertEqual(target_opportunity.call_count, 1)


if __name__ == "__main__":
    unittest.main()
