import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_remix.recommender import rank_candidates


class StubSurrogate:
    def predict_mean(self, rows):
        return [row["measured_gain_hint"] for row in rows]

    def predict_std(self, rows):
        return [0.0 for _ in rows]


class SliceRemixRecommenderTests(unittest.TestCase):
    def test_rank_candidates_prefers_high_gain_low_complexity(self):
        candidates = [
            {"candidate_id": "a", "measured_gain_hint": 0.5, "l1_shift": 0.2, "support_size": 2},
            {"candidate_id": "b", "measured_gain_hint": 0.4, "l1_shift": 0.1, "support_size": 2},
        ]

        ranked = rank_candidates(
            candidates,
            StubSurrogate(),
            kappa=0.0,
            lambda_l1=0.0,
            lambda_support=0.0,
        )

        self.assertEqual(ranked[0]["candidate_id"], "a")


if __name__ == "__main__":
    unittest.main()
