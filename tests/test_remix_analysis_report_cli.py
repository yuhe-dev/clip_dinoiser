import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_remix_analysis_report import build_parser, main


class RemixAnalysisReportCliTests(unittest.TestCase):
    def test_analysis_report_cli_parser_accepts_required_args(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--response-dataset",
                "/tmp/rows_labeled.jsonl",
                "--recommendation-path",
                "/tmp/recommendation.json",
                "--output-path",
                "/tmp/report.json",
            ]
        )

        self.assertEqual(args.response_dataset, "/tmp/rows_labeled.jsonl")
        self.assertEqual(args.recommendation_path, "/tmp/recommendation.json")

    def test_analysis_report_cli_writes_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows_path = os.path.join(tmpdir, "rows_labeled.jsonl")
            recommendation_path = os.path.join(tmpdir, "recommendation.json")
            output_path = os.path.join(tmpdir, "analysis.json")

            rows = [
                {
                    "candidate_id": "cand_0_0",
                    "delta_q": [0.1, -0.1],
                    "delta_phi": {"feature_a": [0.2]},
                    "context": {"baseline_seed": 0},
                    "baseline_metric_value": 24.0,
                    "candidate_metric_value": 24.1,
                    "measured_gain": 0.1,
                },
                {
                    "candidate_id": "cand_0_1",
                    "delta_q": [-0.1, 0.1],
                    "delta_phi": {"feature_a": [-0.2]},
                    "context": {"baseline_seed": 0},
                    "baseline_metric_value": 24.0,
                    "candidate_metric_value": 23.95,
                    "measured_gain": -0.05,
                },
                {
                    "candidate_id": "cand_1_0",
                    "delta_q": [0.1, -0.1],
                    "delta_phi": {"feature_a": [0.2]},
                    "context": {"baseline_seed": 1},
                    "baseline_metric_value": 24.2,
                    "candidate_metric_value": 24.25,
                    "measured_gain": 0.05,
                },
                {
                    "candidate_id": "cand_1_1",
                    "delta_q": [-0.1, 0.1],
                    "delta_phi": {"feature_a": [-0.2]},
                    "context": {"baseline_seed": 1},
                    "baseline_metric_value": 24.2,
                    "candidate_metric_value": 24.15,
                    "measured_gain": -0.05,
                },
            ]
            with open(rows_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            recommendation = {
                "candidate_id": "cand_0_0",
                "predicted_gain_mean": 0.08,
                "predicted_gain_std": 0.02,
                "risk_adjusted_score": 0.06,
                "delta_q": [0.1, -0.1],
                "rationale": {"donors": [1], "receivers": [0]},
                "portrait_summary": {"top_blocks": []},
                "context": {
                    "baseline_seed": 0,
                    "budget": 1000,
                    "surrogate_model": "linear",
                    "bootstrap_models": 1,
                    "kappa": 1.0,
                },
                "ranked_candidates": [
                    {"candidate_id": "cand_0_0", "predicted_gain_mean": 0.08, "predicted_gain_std": 0.02, "risk_adjusted_score": 0.06},
                    {"candidate_id": "cand_0_1", "predicted_gain_mean": -0.02, "predicted_gain_std": 0.01, "risk_adjusted_score": -0.03},
                ],
            }
            with open(recommendation_path, "w", encoding="utf-8") as f:
                json.dump(recommendation, f)

            self.assertEqual(
                main(
                    [
                        "--response-dataset",
                        rows_path,
                        "--recommendation-path",
                        recommendation_path,
                        "--output-path",
                        output_path,
                    ]
                ),
                0,
            )

            with open(output_path, "r", encoding="utf-8") as f:
                report = json.load(f)

            self.assertIn("surrogate", report)
            self.assertIn("recommendation", report)
            self.assertIn("ranked_candidates", report)
            self.assertEqual(report["recommendation"]["actual_comparison"]["source"], "response_dataset")
            self.assertEqual(report["ranked_candidates"][0]["actual_rank_within_baseline"], 1)


if __name__ == "__main__":
    unittest.main()
