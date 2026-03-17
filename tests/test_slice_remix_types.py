import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_remix.types import RecommendationResult


class SliceRemixTypesTests(unittest.TestCase):
    def test_recommendation_result_can_be_constructed(self):
        result = RecommendationResult(
            baseline_mixture=[0.5, 0.5],
            target_mixture=[0.6, 0.4],
            delta_q=[0.1, -0.1],
            predicted_gain_mean=0.2,
            predicted_gain_std=0.05,
            risk_adjusted_score=0.15,
            rationale={},
            execution={},
        )

        self.assertEqual(result.delta_q, [0.1, -0.1])


if __name__ == "__main__":
    unittest.main()
