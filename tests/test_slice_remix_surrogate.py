import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_remix.surrogate import (
    BootstrapRemixSurrogate,
    LinearRemixSurrogate,
    QuadraticRemixSurrogate,
)


class SliceRemixSurrogateTests(unittest.TestCase):
    def test_linear_surrogate_fits_and_predicts(self):
        model = LinearRemixSurrogate()
        rows = [
            {
                "delta_q": [0.1, -0.1],
                "delta_phi": {"feature_a": [0.2]},
                "context": {"budget": 1000},
                "measured_gain": 0.3,
            },
            {
                "delta_q": [-0.1, 0.1],
                "delta_phi": {"feature_a": [-0.2]},
                "context": {"budget": 1000},
                "measured_gain": -0.2,
            },
        ]

        model.fit(rows)
        pred = model.predict_mean(rows[:1])[0]

        self.assertIsInstance(pred, float)
        self.assertEqual(model.predict_std(rows[:1]), [0.0])

    def test_quadratic_surrogate_captures_delta_q_interaction(self):
        model = QuadraticRemixSurrogate()
        rows = [
            {"delta_q": [1.0, 1.0], "delta_phi": {"feature_a": [0.0]}, "context": {"budget": 1000}, "measured_gain": 1.0},
            {"delta_q": [1.0, -1.0], "delta_phi": {"feature_a": [0.0]}, "context": {"budget": 1000}, "measured_gain": -1.0},
            {"delta_q": [-1.0, 1.0], "delta_phi": {"feature_a": [0.0]}, "context": {"budget": 1000}, "measured_gain": -1.0},
            {"delta_q": [-1.0, -1.0], "delta_phi": {"feature_a": [0.0]}, "context": {"budget": 1000}, "measured_gain": 1.0},
        ]

        model.fit(rows)
        preds = model.predict_mean(rows)

        self.assertTrue(all(isinstance(pred, float) for pred in preds))
        for pred, row in zip(preds, rows):
            self.assertAlmostEqual(pred, row["measured_gain"], places=4)

    def test_bootstrap_surrogate_reports_nonzero_uncertainty(self):
        model = BootstrapRemixSurrogate(base_model="linear", num_models=8, seed=3)
        rows = [
            {"delta_q": [0.1, -0.1], "delta_phi": {"feature_a": [0.2]}, "context": {"budget": 1000}, "measured_gain": 0.3},
            {"delta_q": [0.1, -0.1], "delta_phi": {"feature_a": [0.2]}, "context": {"budget": 1000}, "measured_gain": 0.5},
            {"delta_q": [-0.1, 0.1], "delta_phi": {"feature_a": [-0.2]}, "context": {"budget": 1000}, "measured_gain": -0.2},
            {"delta_q": [-0.1, 0.1], "delta_phi": {"feature_a": [-0.2]}, "context": {"budget": 1000}, "measured_gain": -0.4},
        ]

        model.fit(rows)
        means = model.predict_mean(rows[:1])
        stds = model.predict_std(rows[:1])

        self.assertEqual(len(means), 1)
        self.assertEqual(len(stds), 1)
        self.assertGreater(stds[0], 0.0)


if __name__ == "__main__":
    unittest.main()
