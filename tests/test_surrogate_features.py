import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_remix.surrogate_features import (
    build_surrogate_feature_payload,
    compute_focus_coverage_stats,
    compute_hard_mixture,
    compute_soft_mixture,
    flatten_feature_groups,
)


class SurrogateFeaturesTests(unittest.TestCase):
    def test_soft_and_hard_mixtures_are_normalized(self):
        memberships = np.asarray(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.2, 0.8],
            ],
            dtype=np.float32,
        )
        hard_assignment = np.asarray([0, 0, 1], dtype=np.int64)
        sample_indices = [0, 2]

        soft = compute_soft_mixture(memberships, sample_indices)
        hard = compute_hard_mixture(hard_assignment, sample_indices, 2)

        np.testing.assert_allclose(soft, np.asarray([0.55, 0.45], dtype=np.float32))
        np.testing.assert_allclose(hard, np.asarray([0.5, 0.5], dtype=np.float32))

    def test_focus_coverage_stats_counts_presence_per_focus_class(self):
        class_presence = np.asarray(
            [
                [1, 0, 1],
                [0, 1, 0],
                [1, 1, 0],
            ],
            dtype=np.int32,
        )
        stats = compute_focus_coverage_stats(class_presence, [0, 2], focus_class_indices=[0, 1])
        self.assertEqual(stats["focus_class_image_counts"], [2, 1])
        self.assertEqual(stats["focus_class_covered_flags"], [1, 1])
        self.assertEqual(stats["focus_class_covered_count"], 2)

    def test_flatten_feature_groups_uses_labels_when_available(self):
        names, vector = flatten_feature_groups(
            {
                "quality.laplacian": np.asarray([0.1, 0.2], dtype=np.float32),
            },
            feature_label_map={"quality.laplacian": ["hist[0]", "hist[1]"]},
        )
        self.assertEqual(names, ["quality.laplacian.hist[0]", "quality.laplacian.hist[1]"])
        np.testing.assert_allclose(vector, np.asarray([0.1, 0.2], dtype=np.float32))

    def test_build_surrogate_feature_payload_includes_realized_features_and_mixtures(self):
        feature_groups = {
            "quality.laplacian": np.asarray(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6],
                ],
                dtype=np.float32,
            ),
            "coverage.knn_local_density": np.asarray(
                [
                    [0.6, 0.8],
                    [0.5, 0.9],
                    [0.4, 1.0],
                ],
                dtype=np.float32,
            ),
        }
        memberships = np.asarray(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.2, 0.8],
            ],
            dtype=np.float32,
        )
        hard_assignment = np.asarray([0, 0, 1], dtype=np.int64)
        class_presence = np.asarray(
            [
                [1, 0, 1],
                [0, 1, 0],
                [1, 1, 0],
            ],
            dtype=np.int32,
        )

        payload = build_surrogate_feature_payload(
            feature_groups=feature_groups,
            sample_indices=[0, 2],
            memberships=memberships,
            hard_assignment=hard_assignment,
            class_presence=class_presence,
            focus_class_indices=[0, 1],
            feature_label_map={
                "quality.laplacian": ["hist[0]", "hist[1]"],
                "coverage.knn_local_density": ["profile[0]", "profile[1]"],
            },
        )

        self.assertEqual(payload["sample_count"], 2)
        self.assertEqual(len(payload["flat_feature_names"]), 6)
        np.testing.assert_allclose(
            np.asarray(payload["soft_mixture"], dtype=np.float32),
            np.asarray([0.55, 0.45], dtype=np.float32),
        )
        self.assertNotIn("hard_mixture", payload)
        self.assertEqual(payload["focus_coverage"]["focus_class_image_counts"], [2, 1])

    def test_build_surrogate_feature_payload_can_optionally_include_hard_mixture(self):
        feature_groups = {
            "quality.laplacian": np.asarray(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ],
                dtype=np.float32,
            ),
        }
        memberships = np.asarray(
            [
                [0.9, 0.1],
                [0.2, 0.8],
            ],
            dtype=np.float32,
        )
        hard_assignment = np.asarray([0, 1], dtype=np.int64)

        payload = build_surrogate_feature_payload(
            feature_groups=feature_groups,
            sample_indices=[0, 1],
            memberships=memberships,
            hard_assignment=hard_assignment,
            include_hard_mixture=True,
        )

        np.testing.assert_allclose(
            np.asarray(payload["hard_mixture"], dtype=np.float32),
            np.asarray([0.5, 0.5], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
