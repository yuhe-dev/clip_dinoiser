import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_remix_collect_results import build_parser


class RemixCollectResultsCliTests(unittest.TestCase):
    def test_collect_results_cli_parser_accepts_required_args(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--rows-path",
                "/tmp/rows.jsonl",
                "--results-dir",
                "/tmp/results",
                "--output-path",
                "/tmp/result_manifest.jsonl",
            ]
        )

        self.assertEqual(args.rows_path, "/tmp/rows.jsonl")
        self.assertEqual(args.results_dir, "/tmp/results")


if __name__ == "__main__":
    unittest.main()
