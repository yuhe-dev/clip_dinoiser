import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_feature_pipeline import build_argparser


class TestRunFeaturePipelineCLI(unittest.TestCase):
    def test_run_feature_pipeline_cli_accepts_stage_and_dimensions(self):
        parser = build_argparser()
        args = parser.parse_args(["--dimensions", "quality", "difficulty", "--stage", "full"])
        self.assertEqual(args.stage, "full")
        self.assertEqual(args.dimensions, ["quality", "difficulty"])


if __name__ == "__main__":
    unittest.main()
