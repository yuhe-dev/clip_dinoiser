import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_remix_response_dataset import build_parser


class RemixResponseDatasetCliTests(unittest.TestCase):
    def test_response_dataset_cli_parser_accepts_required_args(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--projected-dir",
                "/tmp/projected",
                "--cluster-dir",
                "/tmp/cluster",
                "--output-path",
                "/tmp/rows.jsonl",
                "--budget",
                "1000",
            ]
        )

        self.assertEqual(args.budget, 1000)

    def test_response_dataset_cli_parser_accepts_semantic_portrait_args(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--projected-dir",
                "/tmp/projected",
                "--cluster-dir",
                "/tmp/cluster",
                "--output-path",
                "/tmp/rows.jsonl",
                "--budget",
                "1000",
                "--portrait-source",
                "semantic",
                "--processed-data-root",
                "/tmp/data_feature",
                "--schema-path",
                "/tmp/schema.json",
            ]
        )

        self.assertEqual(args.portrait_source, "semantic")
        self.assertEqual(args.processed_data_root, "/tmp/data_feature")
        self.assertEqual(args.schema_path, "/tmp/schema.json")

    def test_response_dataset_cli_parser_accepts_pair_selector(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--projected-dir",
                "/tmp/projected",
                "--cluster-dir",
                "/tmp/cluster",
                "--output-path",
                "/tmp/rows.jsonl",
                "--budget",
                "1000",
                "--pair-selector",
                "beam_v1",
            ]
        )

        self.assertEqual(args.pair_selector, "beam_v1")

    def test_response_dataset_cli_parser_accepts_target_beam_selector(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--projected-dir",
                "/tmp/projected",
                "--cluster-dir",
                "/tmp/cluster",
                "--output-path",
                "/tmp/rows.jsonl",
                "--budget",
                "1000",
                "--pair-selector",
                "beam_target_v1",
            ]
        )

        self.assertEqual(args.pair_selector, "beam_target_v1")

    def test_response_dataset_cli_parser_accepts_assembled_feature_dir(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--projected-dir",
                "/tmp/projected",
                "--cluster-dir",
                "/tmp/cluster",
                "--output-path",
                "/tmp/rows.jsonl",
                "--budget",
                "1000",
                "--assembled-feature-dir",
                "/tmp/assembled_features",
            ]
        )

        self.assertEqual(args.assembled_feature_dir, "/tmp/assembled_features")

    def test_response_dataset_cli_parser_accepts_coverage_args(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--projected-dir",
                "/tmp/projected",
                "--cluster-dir",
                "/tmp/cluster",
                "--output-path",
                "/tmp/rows.jsonl",
                "--budget",
                "1000",
                "--annotation-root",
                "/tmp/coco_stuff164k",
                "--baseline-result-path",
                "/tmp/baseline.json",
                "--full-result-path",
                "/tmp/full.json",
                "--focus-class-gap-threshold",
                "12.5",
                "--focus-class-top-k",
                "18",
                "--coverage-alpha",
                "0.4",
                "--coverage-repair-budget",
                "96",
            ]
        )

        self.assertEqual(args.annotation_root, "/tmp/coco_stuff164k")
        self.assertEqual(args.baseline_result_path, "/tmp/baseline.json")
        self.assertEqual(args.full_result_path, "/tmp/full.json")
        self.assertEqual(args.focus_class_gap_threshold, 12.5)
        self.assertEqual(args.focus_class_top_k, 18)
        self.assertEqual(args.coverage_alpha, 0.4)
        self.assertEqual(args.coverage_repair_budget, 96)


if __name__ == "__main__":
    unittest.main()
