import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_remix_attach_results import build_parser


class RemixAttachResultsCliTests(unittest.TestCase):
    def test_attach_results_cli_parser_accepts_required_args(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--rows-path",
                "/tmp/rows.jsonl",
                "--result-manifest",
                "/tmp/results.jsonl",
                "--metric-path",
                "coco_stuff.summary.mIoU",
                "--output-path",
                "/tmp/labeled_rows.jsonl",
            ]
        )

        self.assertEqual(args.metric_path, "coco_stuff.summary.mIoU")


if __name__ == "__main__":
    unittest.main()
