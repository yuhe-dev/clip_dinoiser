import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_remix_recommendation import build_parser


class RemixRecommendationCliTests(unittest.TestCase):
    def test_recommendation_cli_parser_accepts_required_args(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--projected-dir",
                "/tmp/projected",
                "--cluster-dir",
                "/tmp/cluster",
                "--response-dataset",
                "/tmp/rows.jsonl",
                "--baseline-seed",
                "0",
                "--budget",
                "1000",
            ]
        )

        self.assertEqual(args.baseline_seed, 0)

    def test_recommendation_cli_parser_accepts_semantic_portrait_args(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--projected-dir",
                "/tmp/projected",
                "--cluster-dir",
                "/tmp/cluster",
                "--response-dataset",
                "/tmp/rows.jsonl",
                "--baseline-seed",
                "0",
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

    def test_recommendation_cli_parser_accepts_surrogate_model_choice(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--projected-dir",
                "/tmp/projected",
                "--cluster-dir",
                "/tmp/cluster",
                "--response-dataset",
                "/tmp/rows.jsonl",
                "--baseline-seed",
                "0",
                "--budget",
                "1000",
                "--surrogate-model",
                "quadratic",
            ]
        )

        self.assertEqual(args.surrogate_model, "quadratic")

    def test_recommendation_cli_parser_accepts_bootstrap_models(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--projected-dir",
                "/tmp/projected",
                "--cluster-dir",
                "/tmp/cluster",
                "--response-dataset",
                "/tmp/rows.jsonl",
                "--baseline-seed",
                "0",
                "--budget",
                "1000",
                "--bootstrap-models",
                "8",
            ]
        )

        self.assertEqual(args.bootstrap_models, 8)

    def test_recommendation_cli_parser_accepts_pair_selector(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--projected-dir",
                "/tmp/projected",
                "--cluster-dir",
                "/tmp/cluster",
                "--response-dataset",
                "/tmp/rows.jsonl",
                "--baseline-seed",
                "0",
                "--budget",
                "1000",
                "--pair-selector",
                "first",
            ]
        )

        self.assertEqual(args.pair_selector, "first")

    def test_recommendation_cli_parser_accepts_assembled_feature_dir(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--projected-dir",
                "/tmp/projected",
                "--cluster-dir",
                "/tmp/cluster",
                "--response-dataset",
                "/tmp/rows.jsonl",
                "--baseline-seed",
                "0",
                "--budget",
                "1000",
                "--assembled-feature-dir",
                "/tmp/assembled_features",
            ]
        )

        self.assertEqual(args.assembled_feature_dir, "/tmp/assembled_features")

    def test_recommendation_cli_parser_accepts_surrogate_output_path(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--projected-dir",
                "/tmp/projected",
                "--cluster-dir",
                "/tmp/cluster",
                "--response-dataset",
                "/tmp/rows.jsonl",
                "--baseline-seed",
                "0",
                "--budget",
                "1000",
                "--surrogate-output-path",
                "/tmp/surrogate.json",
            ]
        )

        self.assertEqual(args.surrogate_output_path, "/tmp/surrogate.json")


if __name__ == "__main__":
    unittest.main()
