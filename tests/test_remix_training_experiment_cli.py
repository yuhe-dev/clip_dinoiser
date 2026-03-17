import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_remix_training_experiment import build_parser


class RemixTrainingExperimentCliTests(unittest.TestCase):
    def test_training_experiment_cli_parser_accepts_required_args(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--config",
                "feature_experiment",
                "--subset-manifest",
                "/tmp/subset.json",
                "--output-dir",
                "/tmp/experiment_out",
                "--result-name",
                "candidate.json",
            ]
        )

        self.assertEqual(args.config, "feature_experiment")
        self.assertEqual(args.result_name, "candidate.json")


if __name__ == "__main__":
    unittest.main()
