import json
import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_search_tree_census import build_parser, run
from clip_dinoiser.slice_remix.dataset import read_jsonl


class SearchTreeCensusCliTests(unittest.TestCase):
    def _write_fixture(self, tmpdir: str) -> tuple[str, str, str]:
        projected_dir = os.path.join(tmpdir, "projected")
        cluster_dir = os.path.join(tmpdir, "cluster")
        os.makedirs(projected_dir, exist_ok=True)
        os.makedirs(cluster_dir, exist_ok=True)

        np.savez(
            os.path.join(projected_dir, "projected_features.npz"),
            matrix=np.asarray(
                [
                    [0.9, 0.1, 0.2, 0.8],
                    [0.8, 0.2, 0.3, 0.7],
                    [0.2, 0.8, 0.7, 0.3],
                    [0.1, 0.9, 0.8, 0.2],
                ],
                dtype=np.float32,
            ),
            sample_ids=np.asarray(["a.jpg", "b.jpg", "c.jpg", "d.jpg"], dtype=object),
        )
        with open(os.path.join(projected_dir, "projected_features_meta.json"), "w", encoding="utf-8") as f:
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
            sample_ids=np.asarray(["a.jpg", "b.jpg", "c.jpg", "d.jpg"], dtype=object),
            membership=np.asarray(
                [
                    [0.9, 0.1, 0.0],
                    [0.8, 0.2, 0.0],
                    [0.1, 0.7, 0.2],
                    [0.0, 0.1, 0.9],
                ],
                dtype=np.float32,
            ),
            hard_assignment=np.asarray([0, 0, 1, 2], dtype=np.int64),
            slice_weights=np.asarray([0.45, 0.275, 0.275], dtype=np.float32),
            centers=np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )
        with open(os.path.join(cluster_dir, "slice_result_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"finder": "vmf", "num_slices": 3}, f)

        schema_path = os.path.join(tmpdir, "schema.json")
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": "processed_feature_bundle.v1",
                    "dimensions": {
                        "quality": {
                            "features": {
                                "laplacian": {
                                    "model_input_fields": ["q50", "q90"],
                                }
                            }
                        },
                        "difficulty": {"features": {}},
                        "coverage": {
                            "features": {
                                "knn_local_density": {
                                    "model_input_fields": ["density_score", "nearest_distance"],
                                }
                            }
                        },
                    },
                },
                f,
            )

        return projected_dir, cluster_dir, schema_path

    def test_parser_accepts_baseline_seeds_and_output_root(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--projected-dir",
                "/tmp/projected",
                "--cluster-dir",
                "/tmp/cluster",
                "--output-root",
                "/tmp/output",
                "--budget",
                "1000",
                "--baseline-seeds",
                "3,11",
            ]
        )

        self.assertEqual(args.output_root, "/tmp/output")
        self.assertEqual(args.baseline_seeds, "3,11")
        self.assertEqual(args.budget, 1000)

    def test_run_exports_session_artifacts_candidate_pool_and_global_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            projected_dir, cluster_dir, schema_path = self._write_fixture(tmpdir)
            output_root = os.path.join(tmpdir, "output")

            parser = build_parser()
            args = parser.parse_args(
                [
                    "--projected-dir",
                    projected_dir,
                    "--cluster-dir",
                    cluster_dir,
                    "--schema-path",
                    schema_path,
                    "--output-root",
                    output_root,
                    "--portrait-source",
                    "projected",
                    "--budget",
                    "2",
                    "--baseline-seeds",
                    "3,11",
                    "--candidate-limit",
                    "4",
                    "--beam-min-transfer-mass",
                    "0.01",
                    "--beam-donor-keep-ratio",
                    "0.1",
                    "--beam-receiver-headroom",
                    "0.3",
                    "--beam-stop-epsilon",
                    "0.0",
                ]
            )

            status = run(args, log_fn=lambda _: None)

            self.assertEqual(status, 0)
            self.assertTrue(os.path.exists(os.path.join(output_root, "feature_label_map.json")))
            self.assertTrue(os.path.exists(os.path.join(output_root, "session_index.json")))
            self.assertTrue(os.path.exists(os.path.join(output_root, "seed_level_summary.json")))
            self.assertTrue(os.path.exists(os.path.join(output_root, "global_coverage_report.json")))
            self.assertTrue(os.path.exists(os.path.join(output_root, "candidate_pool_unlabeled.jsonl")))

            session_dir_seed3 = os.path.join(output_root, "sessions", "baseline_seed_0003")
            session_dir_seed11 = os.path.join(output_root, "sessions", "baseline_seed_0011")
            self.assertTrue(os.path.exists(os.path.join(session_dir_seed3, "session.json")))
            self.assertTrue(os.path.exists(os.path.join(session_dir_seed3, "prior_graph.json")))
            self.assertTrue(os.path.exists(os.path.join(session_dir_seed3, "search_tree.json")))
            self.assertTrue(os.path.exists(os.path.join(session_dir_seed3, "completed_candidates.json")))
            self.assertTrue(os.path.exists(os.path.join(session_dir_seed11, "session.json")))

            with open(os.path.join(output_root, "feature_label_map.json"), "r", encoding="utf-8") as f:
                feature_label_map = json.load(f)
            self.assertIn("quality.laplacian", feature_label_map)
            self.assertIn("coverage.knn_local_density", feature_label_map)

            with open(os.path.join(output_root, "session_index.json"), "r", encoding="utf-8") as f:
                session_index = json.load(f)
            self.assertEqual(len(session_index), 2)
            self.assertEqual(
                {int(row["baseline_seed"]) for row in session_index},
                {3, 11},
            )

            rows = read_jsonl(os.path.join(output_root, "candidate_pool_unlabeled.jsonl"))
            self.assertTrue(rows)
            self.assertEqual({int(row["baseline_seed"]) for row in rows}, {3, 11})

            first_row = rows[0]
            self.assertIn("baseline_trial_id", first_row)
            self.assertIn("baseline_mixture", first_row)
            self.assertIn("target_mixture", first_row)
            self.assertIn("baseline_features_raw", first_row)
            self.assertIn("target_features_raw", first_row)
            self.assertIn("baseline_features_summary", first_row)
            self.assertIn("target_features_summary", first_row)
            self.assertIn("analysis_only", first_row)
            self.assertEqual(len(first_row["baseline_mixture"]), 3)
            self.assertEqual(len(first_row["target_mixture"]), 3)
            self.assertIn("transfer_pairs", first_row)
            self.assertIn("plan_length", first_row)

            with open(os.path.join(output_root, "global_coverage_report.json"), "r", encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(int(report["seed_count"]), 2)
            self.assertEqual(int(report["candidate_pool_size"]), len(rows))
            self.assertGreaterEqual(int(report["total_search_nodes"]), 2)
            self.assertIn("candidate_plan_length_hist", report)
            self.assertIn("unique_primitive_pairs_top_candidates", report)
            self.assertIn("recurrent_primitive_pairs_top_candidates", report)
            self.assertIn("baseline_mixture_dispersion", report)


if __name__ == "__main__":
    unittest.main()
