import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_remix.beam_search import (
    BeamSearchConfig,
    SearchEdge,
    generate_beam_candidates,
    generate_beam_candidates_with_trace,
    get_adaptive_amplitudes,
)
from clip_dinoiser.slice_remix.prior_graph import build_portrait_residual_context


class SliceRemixBeamSearchTests(unittest.TestCase):
    def test_get_adaptive_amplitudes_returns_deduplicated_sorted_steps(self):
        amplitudes = get_adaptive_amplitudes(
            delta_bal=0.08,
            delta_max=0.08 + 1e-8,
            min_transfer_mass=0.03,
        )

        self.assertEqual(len(amplitudes), 2)
        self.assertGreater(amplitudes[0], 0.03)
        self.assertLess(amplitudes[0], 0.08)
        self.assertAlmostEqual(amplitudes[1], 0.08, places=6)

    def test_generate_beam_candidates_preserves_simplex(self):
        baseline = np.asarray([0.55, 0.20, 0.25], dtype=np.float32)
        edges = [
            SearchEdge(donor=0, receiver=1, score=0.9, balance_score=0.8, risk_score=0.1, amplitude_band=(0.03, 0.20)),
            SearchEdge(donor=0, receiver=2, score=0.7, balance_score=0.5, risk_score=0.1, amplitude_band=(0.03, 0.15)),
        ]

        candidates = generate_beam_candidates(
            baseline_mixture=baseline,
            pool_mixture=np.asarray([0.25, 0.35, 0.40], dtype=np.float32),
            edges=edges,
            config=BeamSearchConfig(max_depth=2, beam_width=4, proposal_edges_per_node=2),
        )

        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertAlmostEqual(sum(candidate.target_mixture), 1.0, places=6)
            self.assertTrue(np.all(np.asarray(candidate.target_mixture) >= -1e-8))

    def test_generate_beam_candidates_avoids_immediate_repeat_of_same_edge(self):
        baseline = np.asarray([0.60, 0.10, 0.30], dtype=np.float32)
        edges = [
            SearchEdge(donor=0, receiver=1, score=1.0, balance_score=1.0, risk_score=0.1, amplitude_band=(0.03, 0.25)),
            SearchEdge(donor=0, receiver=2, score=0.8, balance_score=0.7, risk_score=0.1, amplitude_band=(0.03, 0.20)),
        ]

        candidates = generate_beam_candidates(
            baseline_mixture=baseline,
            pool_mixture=np.asarray([0.25, 0.35, 0.40], dtype=np.float32),
            edges=edges,
            config=BeamSearchConfig(max_depth=3, beam_width=6, proposal_edges_per_node=2),
        )

        multi_step = [candidate for candidate in candidates if len(candidate.metadata.get("plan", [])) >= 2]
        self.assertTrue(multi_step)
        for candidate in multi_step:
            plan = candidate.metadata["plan"]
            for left, right in zip(plan, plan[1:]):
                self.assertNotEqual((left["donor"], left["receiver"]), (right["donor"], right["receiver"]))

    def test_generate_beam_candidates_emits_multi_step_sparse_plans(self):
        baseline = np.asarray([0.55, 0.15, 0.15, 0.15], dtype=np.float32)
        edges = [
            SearchEdge(donor=0, receiver=1, score=0.9, balance_score=0.7, risk_score=0.1, amplitude_band=(0.0, 0.02)),
            SearchEdge(donor=0, receiver=2, score=0.85, balance_score=0.6, risk_score=0.1, amplitude_band=(0.0, 0.02)),
            SearchEdge(donor=0, receiver=3, score=0.8, balance_score=0.6, risk_score=0.1, amplitude_band=(0.0, 0.02)),
        ]

        candidates = generate_beam_candidates(
            baseline_mixture=baseline,
            pool_mixture=np.asarray([0.20, 0.30, 0.25, 0.25], dtype=np.float32),
            edges=edges,
            config=BeamSearchConfig(max_depth=3, beam_width=8, proposal_edges_per_node=3),
        )

        self.assertTrue(any(len(candidate.metadata.get("plan", [])) >= 2 for candidate in candidates))

    def test_generate_beam_candidates_accepts_small_feasible_amplitudes_with_adaptive_floor(self):
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
            "quality.score": np.asarray([0.0, 0.0, 2.0, -1.0], dtype=np.float32),
        }
        feature_label_map = {"quality.score": ["score"]}
        portrait_context = build_portrait_residual_context(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
            baseline_sample_indices=[0, 1],
        )
        edges = [
            SearchEdge(donor=0, receiver=1, score=0.8, balance_score=0.7, risk_score=0.1, amplitude_band=(0.0, 0.015)),
        ]

        candidates, trace = generate_beam_candidates_with_trace(
            baseline_mixture=portrait_context.baseline_mixture,
            pool_mixture=portrait_context.pool_mixture,
            edges=edges,
            portrait_context=portrait_context,
            config=BeamSearchConfig(max_depth=2, beam_width=4, proposal_edges_per_node=1),
        )

        self.assertTrue(candidates)
        self.assertEqual(trace["nodes"][0]["depth"], 0)
        self.assertLess(float(candidates[0].metadata["plan"][0]["amplitude"]), 0.03)

    def test_generate_beam_candidates_rejects_non_improving_children_even_with_high_edge_score(self):
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
            "quality.score": np.asarray([0.0, 0.0, 3.0, -3.0], dtype=np.float32),
        }
        feature_label_map = {"quality.score": ["score"]}
        portrait_context = build_portrait_residual_context(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
            baseline_sample_indices=[0, 1],
        )
        edges = [
            SearchEdge(donor=0, receiver=2, score=10.0, balance_score=1.0, risk_score=0.0, amplitude_band=(0.0, 0.10)),
        ]

        candidates = generate_beam_candidates(
            baseline_mixture=portrait_context.baseline_mixture,
            pool_mixture=portrait_context.pool_mixture,
            edges=edges,
            portrait_context=portrait_context,
            config=BeamSearchConfig(max_depth=1, beam_width=4, proposal_edges_per_node=1, lambda_opportunity=1.0),
        )

        self.assertFalse(candidates)

    def test_generate_beam_candidates_prefers_portrait_correct_edge_when_context_provided(self):
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
            "quality.score": np.asarray([0.0, 0.0, 2.0, -1.0], dtype=np.float32),
        }
        feature_label_map = {"quality.score": ["score"]}
        portrait_context = build_portrait_residual_context(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
            baseline_sample_indices=[0, 1],
        )
        edges = [
            SearchEdge(donor=0, receiver=1, score=0.8, balance_score=0.7, risk_score=0.1, amplitude_band=(0.03, 0.20)),
            SearchEdge(donor=0, receiver=2, score=0.8, balance_score=0.7, risk_score=0.1, amplitude_band=(0.03, 0.20)),
        ]

        candidates = generate_beam_candidates(
            baseline_mixture=portrait_context.baseline_mixture,
            pool_mixture=portrait_context.pool_mixture,
            edges=edges,
            portrait_context=portrait_context,
            config=BeamSearchConfig(max_depth=1, beam_width=4, proposal_edges_per_node=2),
        )

        one_step = [candidate for candidate in candidates if len(candidate.metadata.get("plan", [])) == 1]
        self.assertTrue(one_step)
        by_receiver = {int(candidate.metadata["plan"][0]["receiver"]): candidate for candidate in one_step}
        self.assertIn(1, by_receiver)
        self.assertNotIn(2, by_receiver)
        self.assertEqual(int(one_step[0].metadata["plan"][0]["receiver"]), 1)

    def test_generate_beam_candidates_prioritizes_feature_improvement_even_with_narrow_proposals(self):
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
            "quality.score": np.asarray([0.0, 0.0, 2.0, -1.0], dtype=np.float32),
        }
        feature_label_map = {"quality.score": ["score"]}
        portrait_context = build_portrait_residual_context(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
            baseline_sample_indices=[0, 1],
        )
        edges = [
            SearchEdge(donor=0, receiver=2, score=1.0, balance_score=0.8, risk_score=0.1, amplitude_band=(0.03, 0.20)),
            SearchEdge(donor=0, receiver=1, score=0.3, balance_score=0.8, risk_score=0.1, amplitude_band=(0.03, 0.20)),
        ]

        narrow = generate_beam_candidates(
            baseline_mixture=portrait_context.baseline_mixture,
            pool_mixture=portrait_context.pool_mixture,
            edges=edges,
            portrait_context=portrait_context,
            config=BeamSearchConfig(max_depth=1, beam_width=1, proposal_edges_per_node=1),
        )
        wide = generate_beam_candidates(
            baseline_mixture=portrait_context.baseline_mixture,
            pool_mixture=portrait_context.pool_mixture,
            edges=edges,
            portrait_context=portrait_context,
            config=BeamSearchConfig(max_depth=1, beam_width=1, proposal_edges_per_node=2),
        )

        self.assertTrue(narrow)
        self.assertTrue(wide)
        self.assertEqual(int(narrow[0].metadata["plan"][0]["receiver"]), 1)
        self.assertEqual(int(wide[0].metadata["plan"][0]["receiver"]), 1)

    def test_generate_beam_candidates_dedupes_equivalent_delta_q_paths(self):
        baseline = np.asarray([0.60, 0.10, 0.30], dtype=np.float32)
        edges = [
            SearchEdge(donor=0, receiver=1, score=0.7, balance_score=0.8, risk_score=0.1, amplitude_band=(0.03, 0.20)),
            SearchEdge(donor=0, receiver=2, score=0.8, balance_score=0.8, risk_score=0.1, amplitude_band=(0.03, 0.20)),
            SearchEdge(donor=2, receiver=1, score=0.8, balance_score=0.8, risk_score=0.1, amplitude_band=(0.03, 0.20)),
        ]

        candidates = generate_beam_candidates(
            baseline_mixture=baseline,
            pool_mixture=np.asarray([0.30, 0.40, 0.30], dtype=np.float32),
            edges=edges,
            config=BeamSearchConfig(max_depth=2, beam_width=6, proposal_edges_per_node=3),
        )

        delta_keys = {
            tuple(np.round(np.asarray(candidate.delta_q, dtype=np.float32), 6).tolist())
            for candidate in candidates
        }
        self.assertEqual(len(delta_keys), len(candidates))

        direct_net = next(
            candidate
            for candidate in candidates
            if np.allclose(
                np.asarray(candidate.delta_q, dtype=np.float32),
                np.asarray([-0.20, 0.20, 0.0], dtype=np.float32),
                atol=1e-6,
            )
        )
        self.assertEqual(len(direct_net.metadata["plan"]), 1)
        self.assertEqual((direct_net.metadata["plan"][0]["donor"], direct_net.metadata["plan"][0]["receiver"]), (0, 1))


if __name__ == "__main__":
    unittest.main()
