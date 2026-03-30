import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.merge_surrogate_results import build_parser as build_merge_parser
from clip_dinoiser.run_surrogate_batch import build_parser as build_batch_parser


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


if __name__ == "__main__":
    unittest.main()
