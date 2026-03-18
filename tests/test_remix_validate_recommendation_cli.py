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


if __name__ == "__main__":
    unittest.main()
