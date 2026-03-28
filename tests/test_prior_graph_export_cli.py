import json
import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_prior_graph_export import build_parser, run


class PriorGraphExportCliTests(unittest.TestCase):
    def test_parser_accepts_required_arguments(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--projected-dir",
                "/tmp/projected",
                "--cluster-dir",
                "/tmp/cluster",
                "--baseline-seed",
                "3",
                "--budget",
                "2",
                "--output-path",
                "/tmp/prior_graph.json",
            ]
        )

        self.assertEqual(args.baseline_seed, 3)
        self.assertEqual(args.budget, 2)

    def test_run_exports_prior_graph_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            projected_dir = os.path.join(tmpdir, "projected")
            cluster_dir = os.path.join(tmpdir, "cluster")
            output_path = os.path.join(tmpdir, "prior_graph.json")
            os.makedirs(projected_dir, exist_ok=True)
            os.makedirs(cluster_dir, exist_ok=True)

            np.savez(
                os.path.join(projected_dir, "projected_features.npz"),
                matrix=np.asarray(
                    [
                        [0.8, 0.2, 0.9, 0.1],
                        [0.3, 0.7, 0.2, 0.8],
                        [0.1, 0.9, 0.1, 0.9],
                    ],
                    dtype=np.float32,
                ),
                sample_ids=np.asarray(["a.jpg", "b.jpg", "c.jpg"], dtype=object),
            )
            with open(
                os.path.join(projected_dir, "projected_features_meta.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "block_ranges": {
                            "quality.laplacian": [0, 2],
                            "coverage.knn_local_density": [2, 4],
                        }
                    },
                    f,
                )

            np.savez(
                os.path.join(cluster_dir, "slice_result.npz"),
                sample_ids=np.asarray(["a.jpg", "b.jpg", "c.jpg"], dtype=object),
                membership=np.asarray(
                    [
                        [0.9, 0.1],
                        [0.2, 0.8],
                        [0.1, 0.9],
                    ],
                    dtype=np.float32,
                ),
                hard_assignment=np.asarray([0, 1, 1], dtype=np.int64),
                slice_weights=np.asarray([0.4, 0.6], dtype=np.float32),
                centers=np.asarray([[0.0], [1.0]], dtype=np.float32),
            )
            with open(
                os.path.join(cluster_dir, "slice_result_meta.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump({"finder": "gmm", "num_slices": 2}, f)

            parser = build_parser()
            args = parser.parse_args(
                [
                    "--projected-dir",
                    projected_dir,
                    "--cluster-dir",
                    cluster_dir,
                    "--baseline-seed",
                    "0",
                    "--budget",
                    "2",
                    "--output-path",
                    output_path,
                    "--portrait-source",
                    "projected",
                    "--top-k-render",
                    "2",
                ]
            )

            status = run(args, log_fn=lambda _: None)

            self.assertEqual(status, 0)
            self.assertTrue(os.path.exists(output_path))
            with open(output_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(len(payload["nodes"]), 2)
            self.assertEqual(len(payload["edges"]), 2)
            self.assertIn("graph_context", payload)
            self.assertIn("defaults", payload)


if __name__ == "__main__":
    unittest.main()
