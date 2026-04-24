import json
import os
import sys
import tempfile
import unittest
from unittest import mock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.research_harness.learner_ladder import run_learner_sensitivity_ladder


class ResearchLearnerLadderTests(unittest.TestCase):
    def test_learner_ladder_summarizes_regime_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subset_manifest_path = os.path.join(tmpdir, "subset.json")
            with open(subset_manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "candidate_id": "subset_anchor",
                        "sample_ids": ["a.jpg", "b.jpg"],
                        "sample_paths": ["/tmp/a.jpg", "/tmp/b.jpg"],
                    },
                    handle,
                )

            values_by_config = {
                "feature_experiment_fast_cached_slide": 24.28,
                "feature_experiment_fast": 24.30,
                "feature_experiment": 24.35,
            }

            def fake_run(command, cwd, env, stdout, stderr, text, check):
                output_dir = command[command.index("--output-dir") + 1]
                config_name = command[command.index("--config") + 1]
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "coco_stuff": {
                                "summary": {"mIoU": values_by_config[config_name]},
                            }
                        },
                        handle,
                    )

                class Completed:
                    returncode = 0

                return Completed()

            regimes = [
                {"regime_id": "fast_cached_1ep", "config_name": "feature_experiment_fast_cached_slide", "training_seed": 0},
                {"regime_id": "fast_1ep", "config_name": "feature_experiment_fast", "training_seed": 0},
                {"regime_id": "standard_3ep", "config_name": "feature_experiment", "training_seed": 0},
            ]
            with mock.patch("clip_dinoiser.research_harness.learner_ladder.subprocess.run", side_effect=fake_run):
                bundle = run_learner_sensitivity_ladder(
                    experiment_id="EXP-P1-003",
                    subset_manifest_path=subset_manifest_path,
                    output_dir=os.path.join(tmpdir, "out"),
                    metric_name="mIoU",
                    regimes=regimes,
                    python_bin=sys.executable,
                    gpu_id="0",
                    log_fn=lambda _message: None,
                )

            self.assertEqual(bundle.summary["completed_regime_count"], 3)
            self.assertEqual(bundle.summary["best_regime_id"], "standard_3ep")
            self.assertAlmostEqual(bundle.summary["regime_range"], 0.07, places=4)
            self.assertTrue(os.path.exists(bundle.metadata["task_plan_path"]))
            self.assertTrue(os.path.exists(bundle.metadata["progress_markdown_path"]))
            self.assertTrue(os.path.exists(bundle.metadata["handoff_path"]))


if __name__ == "__main__":
    unittest.main()
