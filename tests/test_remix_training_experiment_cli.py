import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_remix_training_experiment import (
    build_parser,
    build_result_entry_filename,
    build_timing_summary,
)


class RemixTrainingExperimentCliTests(unittest.TestCase):
    def test_training_experiment_cli_parser_accepts_required_args(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--config",
                "feature_experiment",
                "--subset-manifest",
                "/tmp/subset.json",
                "--output-dir",
                "/tmp/experiment_out",
                "--result-name",
                "candidate.json",
            ]
        )

        self.assertEqual(args.config, "feature_experiment")
        self.assertEqual(args.result_name, "candidate.json")

    def test_build_timing_summary_reports_stage_and_total_seconds(self):
        timing = build_timing_summary(
            train_seconds=12.3456,
            eval_seconds=3.2109,
            total_seconds=15.5565,
            subset_size=1000,
            started_at="2026-03-18T10:00:00+00:00",
            finished_at="2026-03-18T10:00:16+00:00",
        )

        self.assertEqual(timing["subset_size"], 1000)
        self.assertEqual(timing["started_at"], "2026-03-18T10:00:00+00:00")
        self.assertEqual(timing["finished_at"], "2026-03-18T10:00:16+00:00")
        self.assertAlmostEqual(timing["train_seconds"], 12.346, places=3)
        self.assertAlmostEqual(timing["eval_seconds"], 3.211, places=3)
        self.assertAlmostEqual(timing["total_seconds"], 15.556, places=3)

    def test_build_result_entry_filename_uses_candidate_id(self):
        self.assertEqual(
            build_result_entry_filename("baseline_0"),
            "baseline_0_result_entry.json",
        )


if __name__ == "__main__":
    unittest.main()
