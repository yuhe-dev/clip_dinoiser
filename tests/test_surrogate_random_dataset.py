import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_surrogate_random_dataset import build_parser, build_split_assignments, parse_seed_spec


class SurrogateRandomDatasetTests(unittest.TestCase):
    def test_parse_seed_spec_supports_ranges_and_singletons(self):
        self.assertEqual(parse_seed_spec("0:3,5,8:10"), [0, 1, 2, 5, 8, 9])

    def test_build_split_assignments_rejects_overlap(self):
        with self.assertRaises(ValueError):
            build_split_assignments(
                train_seeds=[0, 1],
                val_seeds=[1, 2],
                test_seeds=[3],
            )

    def test_cli_parser_accepts_random_dataset_args(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--projected-dir",
                "/tmp/projected",
                "--cluster-dir",
                "/tmp/cluster",
                "--output-path",
                "/tmp/out.jsonl",
                "--budget",
                "1000",
                "--train-seeds",
                "0:10",
                "--val-seeds",
                "10:12",
                "--test-seeds",
                "12:15",
                "--training-seed",
                "7",
                "--include-hard-mixture",
            ]
        )
        self.assertEqual(args.budget, 1000)
        self.assertEqual(args.train_seeds, "0:10")
        self.assertEqual(args.training_seed, 7)
        self.assertTrue(args.include_hard_mixture)


if __name__ == "__main__":
    unittest.main()
