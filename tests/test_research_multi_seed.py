import json
import os
import sys
import tempfile
import unittest
from unittest import mock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.research_harness.multi_seed import run_same_subset_multi_seed


class ResearchMultiSeedTests(unittest.TestCase):
    def test_same_subset_multi_seed_summarizes_completed_runs(self):
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

            values_by_seed = {0: 24.20, 1: 24.30, 2: 24.40}

            def fake_run(command, cwd, env, stdout, stderr, text, check):
                output_dir = command[command.index("--output-dir") + 1]
                seed = int(command[command.index("--seed") + 1])
                os.makedirs(output_dir, exist_ok=True)
                result_path = os.path.join(output_dir, "result.json")
                with open(result_path, "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "coco_stuff": {
                                "summary": {"mIoU": values_by_seed[seed]},
                            }
                        },
                        handle,
                    )

                class Completed:
                    returncode = 0

                return Completed()

            with mock.patch("clip_dinoiser.research_harness.multi_seed.subprocess.run", side_effect=fake_run):
                bundle = run_same_subset_multi_seed(
                    experiment_id="EXP-P1-002",
                    subset_manifest_path=subset_manifest_path,
                    output_dir=os.path.join(tmpdir, "out"),
                    metric_name="mIoU",
                    config_name="feature_experiment_fast_cached_slide",
                    training_seeds=[0, 1, 2],
                    python_bin=sys.executable,
                    gpu_id="0",
                    log_fn=lambda _message: None,
                )

            self.assertEqual(bundle.summary["completed_seed_count"], 3)
            self.assertAlmostEqual(bundle.summary["mean"], 24.30)
            self.assertAlmostEqual(bundle.summary["range"], 0.20)
            self.assertEqual(bundle.summary["training_seed_values"], [0, 1, 2])
            self.assertTrue(os.path.exists(bundle.metadata["progress_path"]))
            self.assertTrue(os.path.exists(bundle.metadata["task_plan_path"]))
            self.assertTrue(os.path.exists(bundle.metadata["progress_markdown_path"]))
            self.assertTrue(os.path.exists(bundle.metadata["handoff_path"]))

    def test_same_subset_multi_seed_reruns_when_result_exists_without_completion_sentinel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subset_manifest_path = os.path.join(tmpdir, "subset.json")
            with open(subset_manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "candidate_id": "subset_anchor",
                        "sample_ids": ["a.jpg"],
                        "sample_paths": ["/tmp/a.jpg"],
                    },
                    handle,
                )

            run_dir = os.path.join(tmpdir, "out", "runs", "subset_anchor_trainseed00")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as handle:
                json.dump({"coco_stuff": {"summary": {"mIoU": 24.00}}}, handle)

            calls = {"count": 0}

            def fake_run(command, cwd, env, stdout, stderr, text, check):
                calls["count"] += 1
                output_dir = command[command.index("--output-dir") + 1]
                with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as handle:
                    json.dump({"coco_stuff": {"summary": {"mIoU": 24.25}}}, handle)

                class Completed:
                    returncode = 0

                return Completed()

            with mock.patch("clip_dinoiser.research_harness.multi_seed.subprocess.run", side_effect=fake_run):
                bundle = run_same_subset_multi_seed(
                    experiment_id="EXP-P1-002",
                    subset_manifest_path=subset_manifest_path,
                    output_dir=os.path.join(tmpdir, "out"),
                    metric_name="mIoU",
                    config_name="feature_experiment_fast_cached_slide",
                    training_seeds=[0],
                    python_bin=sys.executable,
                    gpu_id="0",
                    log_fn=lambda _message: None,
                )

            self.assertEqual(calls["count"], 1)
            self.assertEqual(bundle.summary["completed_seed_count"], 1)
            self.assertTrue(os.path.exists(os.path.join(run_dir, "completion.json")))

    def test_same_subset_multi_seed_reruns_when_completion_sentinel_provenance_mismatches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subset_manifest_path = os.path.join(tmpdir, "subset.json")
            with open(subset_manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "candidate_id": "subset_anchor",
                        "sample_ids": ["a.jpg"],
                        "sample_paths": ["/tmp/a.jpg"],
                    },
                    handle,
                )

            run_dir = os.path.join(tmpdir, "out", "runs", "subset_anchor_trainseed00")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as handle:
                json.dump({"coco_stuff": {"summary": {"mIoU": 24.00}}}, handle)
            with open(os.path.join(run_dir, "completion.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-P1-002",
                        "candidate_id": "subset_anchor_trainseed00",
                        "training_seed": 0,
                        "metric_name": "mIoU",
                        "python_bin": "/wrong/python",
                        "runtime_profile_id": "base",
                        "config_name": "old_config",
                        "subset_manifest_path": "/wrong/subset.json",
                    },
                    handle,
                )

            calls = {"count": 0}

            def fake_run(command, cwd, env, stdout, stderr, text, check):
                calls["count"] += 1
                output_dir = command[command.index("--output-dir") + 1]
                with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as handle:
                    json.dump({"coco_stuff": {"summary": {"mIoU": 24.35}}}, handle)

                class Completed:
                    returncode = 0

                return Completed()

            with mock.patch("clip_dinoiser.research_harness.multi_seed.subprocess.run", side_effect=fake_run):
                bundle = run_same_subset_multi_seed(
                    experiment_id="EXP-P1-002",
                    subset_manifest_path=subset_manifest_path,
                    output_dir=os.path.join(tmpdir, "out"),
                    metric_name="mIoU",
                    config_name="feature_experiment_fast_cached_slide",
                    training_seeds=[0],
                    python_bin=sys.executable,
                    runtime_profile_id="clipdino2",
                    gpu_id="0",
                    log_fn=lambda _message: None,
                )

            self.assertEqual(calls["count"], 1)
            self.assertEqual(bundle.summary["completed_seed_count"], 1)

    def test_same_subset_multi_seed_reuses_legacy_completion_sentinel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subset_manifest_path = os.path.join(tmpdir, "subset.json")
            with open(subset_manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "candidate_id": "subset_anchor",
                        "sample_ids": ["a.jpg"],
                        "sample_paths": ["/tmp/a.jpg"],
                    },
                    handle,
                )

            run_dir = os.path.join(tmpdir, "out", "runs", "subset_anchor_trainseed00")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as handle:
                json.dump({"coco_stuff": {"summary": {"mIoU": 24.31}}}, handle)
            with open(os.path.join(run_dir, "completion.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-P1-002",
                        "candidate_id": "subset_anchor_trainseed00",
                        "training_seed": 0,
                        "metric_name": "mIoU",
                        "python_bin": sys.executable,
                    },
                    handle,
                )

            calls = {"count": 0}

            def fake_run(command, cwd, env, stdout, stderr, text, check):
                calls["count"] += 1
                raise AssertionError("legacy completion should have been reused")

            with mock.patch("clip_dinoiser.research_harness.multi_seed.subprocess.run", side_effect=fake_run):
                bundle = run_same_subset_multi_seed(
                    experiment_id="EXP-P1-002",
                    subset_manifest_path=subset_manifest_path,
                    output_dir=os.path.join(tmpdir, "out"),
                    metric_name="mIoU",
                    config_name="feature_experiment_fast_cached_slide",
                    training_seeds=[0],
                    python_bin=sys.executable,
                    runtime_profile_id="clipdino2",
                    gpu_id="0",
                    log_fn=lambda _message: None,
                )

            self.assertEqual(calls["count"], 0)
            self.assertEqual(bundle.summary["completed_seed_count"], 1)


if __name__ == "__main__":
    unittest.main()
