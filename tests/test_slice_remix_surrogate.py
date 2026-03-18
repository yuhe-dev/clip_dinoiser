import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_remix.surrogate import (
    BootstrapRemixSurrogate,
    LinearRemixSurrogate,
    QuadraticRemixSurrogate,
    cross_validate_surrogate,
    evaluate_surrogate_predictions,
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

    def test_linear_surrogate_serializes_to_json(self):
        model = LinearRemixSurrogate()
        rows = [
            {"delta_q": [0.1, -0.1], "delta_phi": {"feature_a": [0.2]}, "context": {"budget": 1000}, "measured_gain": 0.3},
            {"delta_q": [-0.1, 0.1], "delta_phi": {"feature_a": [-0.2]}, "context": {"budget": 1000}, "measured_gain": -0.2},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "surrogate.json")
            model.fit(rows).save_json(output_path)

            self.assertTrue(os.path.isfile(output_path))

    def test_evaluate_surrogate_predictions_reports_group_metrics(self):
        rows = [
            {"candidate_id": "cand_0_0", "delta_q": [0.1, -0.1], "delta_phi": {"feature_a": [0.2]}, "context": {"baseline_seed": 0}, "measured_gain": 0.2},
            {"candidate_id": "cand_0_1", "delta_q": [0.0, 0.0], "delta_phi": {"feature_a": [0.0]}, "context": {"baseline_seed": 0}, "measured_gain": -0.1},
            {"candidate_id": "cand_1_0", "delta_q": [0.2, -0.2], "delta_phi": {"feature_a": [0.4]}, "context": {"baseline_seed": 1}, "measured_gain": 0.5},
            {"candidate_id": "cand_1_1", "delta_q": [-0.2, 0.2], "delta_phi": {"feature_a": [-0.4]}, "context": {"baseline_seed": 1}, "measured_gain": -0.3},
        ]

        report = evaluate_surrogate_predictions(
            rows,
            predicted_mean=[0.18, -0.05, 0.45, -0.2],
            predicted_std=[0.01, 0.01, 0.02, 0.02],
            group_key="baseline_seed",
            top_k=1,
            kappa=1.0,
        )

        self.assertIn("metrics", report)
        self.assertAlmostEqual(report["metrics"]["top1_hit_rate"], 1.0)
        self.assertEqual(len(report["group_reports"]), 2)

    def test_cross_validate_surrogate_runs_leave_one_group_out(self):
        rows = [
            {"candidate_id": "cand_0_0", "delta_q": [1.0, 0.0], "delta_phi": {"feature_a": [1.0]}, "context": {"baseline_seed": 0}, "measured_gain": 1.0},
            {"candidate_id": "cand_0_1", "delta_q": [0.0, 1.0], "delta_phi": {"feature_a": [-1.0]}, "context": {"baseline_seed": 0}, "measured_gain": -1.0},
            {"candidate_id": "cand_1_0", "delta_q": [1.0, 0.0], "delta_phi": {"feature_a": [1.0]}, "context": {"baseline_seed": 1}, "measured_gain": 1.0},
            {"candidate_id": "cand_1_1", "delta_q": [0.0, 1.0], "delta_phi": {"feature_a": [-1.0]}, "context": {"baseline_seed": 1}, "measured_gain": -1.0},
        ]

        report = cross_validate_surrogate(
            rows,
            model_name="linear",
            bootstrap_models=1,
            group_key="baseline_seed",
            top_k=1,
            kappa=0.0,
        )

        self.assertEqual(report["mode"], "leave_one_group_out")
        self.assertEqual(report["group_count"], 2)
        self.assertEqual(len(report["predictions"]), 4)


if __name__ == "__main__":
    unittest.main()
