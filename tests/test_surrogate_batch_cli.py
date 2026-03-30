import os
import json
import sys
import tempfile
import unittest
from unittest import mock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.merge_surrogate_results import build_parser as build_merge_parser
from clip_dinoiser.run_surrogate_batch import build_parser as build_batch_parser
import clip_dinoiser.run_surrogate_batch as surrogate_batch


class SurrogateBatchCliTests(unittest.TestCase):
    def test_batch_cli_parser_accepts_parallel_runner_args(self):
        parser = build_batch_parser()
        args = parser.parse_args(
            [
                "--dataset-jsonl",
                "/tmp/random_subsets.jsonl",
                "--runs-root",
                "/tmp/runs",
                "--config",
                "feature_experiment_fast_cached_slide",
                "--gpus",
                "0,1,2,3",
                "--master-port-base",
                "29600",
                "--split",
                "train",
                "--limit",
                "16",
            ]
        )
        self.assertEqual(args.split, "train")
        self.assertEqual(args.limit, 16)
        self.assertEqual(args.master_port_base, 29600)

    def test_merge_cli_parser_accepts_required_args(self):
        parser = build_merge_parser()
        args = parser.parse_args(
            [
                "--dataset-jsonl",
                "/tmp/random_subsets.jsonl",
                "--runs-root",
                "/tmp/runs",
                "--output-path",
                "/tmp/merged.jsonl",
            ]
        )
        self.assertEqual(args.output_path, "/tmp/merged.jsonl")

    def test_runner_reuses_first_free_slot_after_earlier_job_finishes(self):
        rows = [
            {
                "experiment_id": "exp0",
                "manifest_path": "/tmp/exp0.json",
                "training_seed": 0,
                "split": "train",
            },
            {
                "experiment_id": "exp1",
                "manifest_path": "/tmp/exp1.json",
                "training_seed": 0,
                "split": "train",
            },
            {
                "experiment_id": "exp2",
                "manifest_path": "/tmp/exp2.json",
                "training_seed": 0,
                "split": "train",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_jsonl = os.path.join(tmpdir, "rows.jsonl")
            with open(dataset_jsonl, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            tick = {"value": 0}
            launches = []

            class FakeProcess:
                def __init__(self, finish_at: int):
                    self.finish_at = finish_at

                def poll(self):
                    if tick["value"] >= self.finish_at:
                        return 0
                    return None

            finish_schedule = [1, 3, 4]

            def fake_popen(command, cwd, env, stdout, stderr):
                launch_index = len(launches)
                launches.append(
                    {
                        "gpu": env["CUDA_VISIBLE_DEVICES"],
                        "port": command[command.index("--master_port") + 1],
                        "output_dir": command[command.index("--output-dir") + 1],
                    }
                )
                return FakeProcess(finish_schedule[launch_index])

            def fake_sleep(_seconds):
                tick["value"] += 1

            args = build_batch_parser().parse_args(
                [
                    "--dataset-jsonl",
                    dataset_jsonl,
                    "--runs-root",
                    os.path.join(tmpdir, "runs"),
                    "--gpus",
                    "0,1",
                    "--master-port-base",
                    "29600",
                ]
            )
            with mock.patch.object(surrogate_batch.subprocess, "Popen", side_effect=fake_popen):
                with mock.patch.object(surrogate_batch.time, "sleep", side_effect=fake_sleep):
                    result = surrogate_batch.run(args, log_fn=lambda _message: None)

        self.assertEqual(result, 0)
        self.assertEqual(
            [(launch["gpu"], launch["port"]) for launch in launches],
            [("0", "29600"), ("1", "29601"), ("0", "29600")],
        )


if __name__ == "__main__":
    unittest.main()
