import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_remix_validate_recommendation import build_parser


class RemixValidateRecommendationCliTests(unittest.TestCase):
    def test_validate_recommendation_cli_parser_accepts_required_args(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--cluster-dir",
                "/tmp/cluster",
                "--recommendation-path",
                "/tmp/recommendation.json",
                "--pool-image-root",
                "/tmp/images",
                "--output-manifest",
                "/tmp/recommended_manifest.json",
            ]
        )

        self.assertEqual(args.cluster_dir, "/tmp/cluster")
        self.assertEqual(args.output_manifest, "/tmp/recommended_manifest.json")

    def test_validate_recommendation_cli_parser_accepts_coverage_args(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--cluster-dir",
                "/tmp/cluster",
                "--recommendation-path",
                "/tmp/recommendation.json",
                "--pool-image-root",
                "/tmp/images",
                "--output-manifest",
                "/tmp/recommended_manifest.json",
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
