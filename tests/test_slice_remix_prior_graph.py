import math
import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_remix.prior_graph import (
    PriorGraphHyperparams,
    PriorGraphUserIntent,
    build_prior_graph,
    build_portrait_residual_context,
)


class SliceRemixPriorGraphTests(unittest.TestCase):
    def _build_fixture(self):
        feature_groups = {
            "quality.laplacian": np.asarray(
                [
                    [0.80, 0.20, 0.90, 0.75],
                    [0.35, 0.65, 0.45, 0.30],
                    [0.10, 0.90, 0.15, 0.10],
                    [0.70, 0.30, 0.70, 0.60],
                ],
                dtype=np.float32,
            ),
            "coverage.knn_local_density": np.asarray(
                [
                    [0.90, 0.10, 0.85],
                    [0.55, 0.45, 0.55],
                    [0.15, 0.85, 0.20],
                    [0.75, 0.25, 0.70],
                ],
                dtype=np.float32,
            ),
        }
        feature_label_map = {
            "quality.laplacian": ["hist[0]", "hist[1]", "q50", "density_score"],
            "coverage.knn_local_density": ["profile[0]", "profile[1]", "density_score"],
        }
        memberships = np.asarray(
            [
                [0.95, 0.03, 0.02],
                [0.08, 0.88, 0.04],
                [0.03, 0.05, 0.92],
                [0.70, 0.20, 0.10],
            ],
            dtype=np.float32,
        )
        return feature_groups, feature_label_map, memberships

    def test_build_prior_graph_masks_frozen_slice_and_marks_visible_edges(self):
        feature_groups, feature_label_map, memberships = self._build_fixture()

        payload = build_prior_graph(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
            baseline_sample_indices=[0, 1],
            slice_ids=["slice_0", "slice_1", "slice_2"],
            user_intent=PriorGraphUserIntent(frozen_slices={"slice_1"}),
            hyperparams=PriorGraphHyperparams(top_k_render=1, score_threshold=-1.0),
        )

        self.assertEqual(len(payload.nodes), 3)
        self.assertEqual(len(payload.edges), 6)

        masked_edges = [
            edge for edge in payload.edges if edge.masked_reason == "frozen_slice"
        ]
        self.assertTrue(masked_edges)
        self.assertTrue(all(not edge.admissible for edge in masked_edges))

        visible_edges = [edge for edge in payload.edges if edge.visible_by_default]
        self.assertEqual(len(visible_edges), 1)
        self.assertTrue(visible_edges[0].admissible)

    def test_build_prior_graph_keeps_edge_scores_json_finite(self):
        feature_groups, feature_label_map, memberships = self._build_fixture()

        payload = build_prior_graph(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
            baseline_sample_indices=[0, 1],
            slice_ids=["slice_0", "slice_1", "slice_2"],
            user_intent=PriorGraphUserIntent(frozen_slices={"slice_1"}),
            hyperparams=PriorGraphHyperparams(top_k_render=6, score_threshold=-1.0),
        )

        self.assertTrue(payload.edges)
        self.assertTrue(all(math.isfinite(edge.score) for edge in payload.edges))

    def test_prior_graph_uses_reduced_parameter_contract(self):
        params = PriorGraphHyperparams()

        self.assertTrue(hasattr(params, "lambda_balance"))
        self.assertTrue(hasattr(params, "lambda_user"))
        self.assertTrue(hasattr(params, "lambda_risk"))
        self.assertTrue(hasattr(params, "shape_rho"))
        self.assertAlmostEqual(params.shape_rho, 0.8, places=6)
        self.assertFalse(hasattr(params, "alpha_fit"))
        self.assertFalse(hasattr(params, "alpha_bal"))
        self.assertFalse(hasattr(params, "beta_side"))

    def test_build_portrait_residual_context_biases_mixed_blocks_toward_shape(self):
        feature_groups, feature_label_map, memberships = self._build_fixture()

        context = build_portrait_residual_context(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
            baseline_sample_indices=[0, 1],
        )

        by_name = {block.layout.name: block for block in context.block_contexts}
        self.assertAlmostEqual(by_name["quality.laplacian"].rho, 0.8, places=6)
        self.assertAlmostEqual(by_name["coverage.knn_local_density"].rho, 0.8, places=6)

    def test_build_prior_graph_reports_balance_but_excludes_it_from_total_score(self):
        feature_groups, feature_label_map, memberships = self._build_fixture()
        params = PriorGraphHyperparams(top_k_render=6, score_threshold=-1.0)

        payload = build_prior_graph(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
            baseline_sample_indices=[0, 1],
            slice_ids=["slice_0", "slice_1", "slice_2"],
            hyperparams=params,
        )

        edge_map = {(edge.donor, edge.receiver): edge for edge in payload.edges}
        toward_underweighted = edge_map[("slice_0", "slice_2")]
        reverse_edge = edge_map[("slice_2", "slice_0")]

        self.assertGreater(toward_underweighted.balance_score, reverse_edge.balance_score)
        for edge in payload.edges:
            expected = edge.fit_score + params.lambda_user * edge.user_score - params.lambda_risk * edge.risk_score
            self.assertAlmostEqual(edge.score, expected, places=6)

    def test_build_prior_graph_reports_fit_user_and_risk_breakdowns(self):
        feature_groups, feature_label_map, memberships = self._build_fixture()

        payload = build_prior_graph(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
            baseline_sample_indices=[0, 1],
            slice_ids=["slice_0", "slice_1", "slice_2"],
            user_intent=PriorGraphUserIntent(
                protected_atomic_blocks={"coverage.knn_local_density"},
                preferred_edges={("slice_0", "slice_2")},
            ),
            hyperparams=PriorGraphHyperparams(top_k_render=6, score_threshold=-1.0),
        )

        edge = next(edge for edge in payload.edges if edge.donor == "slice_0" and edge.receiver == "slice_2")

        self.assertIn("quality.laplacian", edge.block_scores)
        self.assertIn("coverage.knn_local_density", edge.block_scores)
        self.assertGreater(edge.fit_score, 0.0)
        self.assertGreater(edge.user_score, 0.0)
        self.assertGreaterEqual(edge.risk_score, 0.0)
        self.assertGreaterEqual(edge.amplitude_band[1], edge.amplitude_band[0])

    def test_prior_graph_defaults_export_reduced_names(self):
        feature_groups, feature_label_map, memberships = self._build_fixture()

        payload = build_prior_graph(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
            baseline_sample_indices=[0, 1],
            slice_ids=["slice_0", "slice_1", "slice_2"],
        )

        self.assertIn("lambda_balance", payload.defaults)
        self.assertIn("lambda_user", payload.defaults)
        self.assertIn("lambda_risk", payload.defaults)
        self.assertNotIn("alpha_fit", payload.defaults)
        self.assertNotIn("alpha_bal", payload.defaults)


if __name__ == "__main__":
    unittest.main()
