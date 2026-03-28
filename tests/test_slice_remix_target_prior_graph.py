import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_remix.prior_graph import (
    PriorGraphHyperparams,
    SearchBias,
    SearchConstraints,
    TargetPortraitSpec,
    build_pool_target_portrait_spec,
    build_target_prior_graph,
    build_target_residual_context,
    compute_target_residual_gap,
)


class SliceRemixTargetPriorGraphTests(unittest.TestCase):
    def _build_fixture(self):
        feature_groups = {
            "quality.laplacian": np.asarray(
                [
                    [0.80, 0.20, 0.10, 0.0, 0.0],
                    [0.70, 0.30, 0.20, 0.0, 0.0],
                    [0.15, 0.35, 0.50, 0.1, 0.0],
                    [0.10, 0.20, 0.70, 0.2, 0.0],
                ],
                dtype=np.float32,
            ),
            "coverage.knn_local_density": np.asarray(
                [
                    [0.80, 0.20, 2.0, 0.0],
                    [0.75, 0.25, 1.8, 0.0],
                    [0.25, 0.75, 0.3, 1.0],
                    [0.20, 0.80, 0.1, 1.0],
                ],
                dtype=np.float32,
            ),
        }
        feature_label_map = {
            "quality.laplacian": ["hist[0]", "hist[1]", "hist[2]", "log_num_values", "empty_flag"],
            "coverage.knn_local_density": ["profile[0]", "profile[1]", "log_num_values", "empty_flag"],
        }
        memberships = np.asarray(
            [
                [0.95, 0.03, 0.02],
                [0.90, 0.05, 0.05],
                [0.05, 0.85, 0.10],
                [0.02, 0.08, 0.90],
            ],
            dtype=np.float32,
        )
        slice_ids = ["slice_00", "slice_01", "slice_02"]
        return feature_groups, feature_label_map, memberships, slice_ids

    def test_target_prior_graph_changes_edge_preference_when_target_changes(self):
        feature_groups, feature_label_map, memberships, slice_ids = self._build_fixture()
        pool_target = build_pool_target_portrait_spec(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
        )
        middle_shift = np.asarray(pool_target.shape_targets["quality.laplacian"], dtype=np.float32).copy()
        middle_shift[0] = max(0.0, middle_shift[0] - 0.18)
        middle_shift[1] += 0.18
        middle_shift = middle_shift / middle_shift.sum()
        middle_target = TargetPortraitSpec(
            shape_targets={"quality.laplacian": middle_shift},
            scalar_targets=dict(pool_target.scalar_targets),
            block_weights={"quality.laplacian": 1.0},
        )
        high_shift = np.asarray(pool_target.shape_targets["quality.laplacian"], dtype=np.float32).copy()
        high_shift[0] = max(0.0, high_shift[0] - 0.18)
        high_shift[2] += 0.18
        high_shift = high_shift / high_shift.sum()
        high_target = TargetPortraitSpec(
            shape_targets={"quality.laplacian": high_shift},
            scalar_targets=dict(pool_target.scalar_targets),
            block_weights={"quality.laplacian": 1.0},
        )

        middle_payload = build_target_prior_graph(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
            baseline_sample_indices=[0, 1],
            slice_ids=slice_ids,
            target_spec=middle_target,
            constraints=SearchConstraints(),
            bias=SearchBias(),
            hyperparams=PriorGraphHyperparams(top_k_render=6, score_threshold=-10.0),
        )
        high_payload = build_target_prior_graph(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
            baseline_sample_indices=[0, 1],
            slice_ids=slice_ids,
            target_spec=high_target,
            constraints=SearchConstraints(),
            bias=SearchBias(),
            hyperparams=PriorGraphHyperparams(top_k_render=6, score_threshold=-10.0),
        )

        middle_admissible = sorted(
            [edge for edge in middle_payload.edges if edge.admissible],
            key=lambda edge: edge.score,
            reverse=True,
        )
        high_admissible = sorted(
            [edge for edge in high_payload.edges if edge.admissible],
            key=lambda edge: edge.score,
            reverse=True,
        )
        self.assertTrue(middle_admissible)
        self.assertTrue(high_admissible)
        middle_scores = {(edge.donor, edge.receiver): edge.score for edge in middle_admissible}
        high_scores = {(edge.donor, edge.receiver): edge.score for edge in high_admissible}
        self.assertGreater(
            middle_scores[("slice_00", "slice_01")],
            high_scores[("slice_00", "slice_01")],
        )
        self.assertGreater(
            high_scores[("slice_00", "slice_02")],
            middle_scores[("slice_00", "slice_02")],
        )

    def test_build_pool_target_portrait_spec_includes_shape_and_scalar_targets(self):
        feature_groups, feature_label_map, memberships, _slice_ids = self._build_fixture()

        target = build_pool_target_portrait_spec(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
        )

        self.assertIn("quality.laplacian", target.shape_targets)
        self.assertIn("quality.laplacian", target.scalar_targets)
        self.assertIn("coverage.knn_local_density", target.shape_targets)
        self.assertIn("coverage.knn_local_density", target.scalar_targets)

    def test_target_residual_gap_changes_when_scalar_target_changes(self):
        feature_groups, feature_label_map, memberships, _slice_ids = self._build_fixture()

        target = build_pool_target_portrait_spec(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
        )
        baseline_context = build_target_residual_context(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
            baseline_sample_indices=[0, 1],
            target_spec=target,
        )
        shifted_scalar_targets = dict(target.scalar_targets)
        shifted_scalar_targets["coverage.knn_local_density"] = (
            np.asarray(shifted_scalar_targets["coverage.knn_local_density"], dtype=np.float32) + 0.75
        ).astype(np.float32)
        shifted_context = build_target_residual_context(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
            baseline_sample_indices=[0, 1],
            target_spec=TargetPortraitSpec(
                shape_targets=dict(target.shape_targets),
                scalar_targets=shifted_scalar_targets,
                block_weights=dict(target.block_weights),
                source=target.source,
            ),
        )

        baseline_gap = compute_target_residual_gap(
            context=baseline_context,
            mixture=baseline_context.baseline_mixture,
        )
        shifted_gap = compute_target_residual_gap(
            context=shifted_context,
            mixture=shifted_context.baseline_mixture,
        )
        self.assertNotAlmostEqual(baseline_gap, shifted_gap, places=6)

    def test_target_prior_graph_exports_target_semantic_defaults(self):
        feature_groups, feature_label_map, memberships, slice_ids = self._build_fixture()
        target = build_pool_target_portrait_spec(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
        )

        payload = build_target_prior_graph(
            feature_groups=feature_groups,
            feature_label_map=feature_label_map,
            memberships=memberships,
            baseline_sample_indices=[0, 1],
            slice_ids=slice_ids,
            target_spec=target,
            constraints=SearchConstraints(),
            bias=SearchBias(),
        )

        self.assertIn("lambda_risk", payload.defaults)
        self.assertNotIn("lambda_balance", payload.defaults)
        self.assertNotIn("lambda_user", payload.defaults)
        edge = next(edge for edge in payload.edges if edge.donor != edge.receiver)
        self.assertTrue(hasattr(edge, "bias_score"))
        self.assertTrue(hasattr(edge, "risk_components"))
        self.assertIn("support_empty_risk", edge.risk_components)
        self.assertNotIn("instability_risk", edge.risk_components)


if __name__ == "__main__":
    unittest.main()
