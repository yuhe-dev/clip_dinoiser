import json
import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_workbench_beam_export import build_parser, run


class WorkbenchBeamExportCliTests(unittest.TestCase):
    def test_run_exports_prior_graph_and_beam_rounds_without_surrogate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            projected_dir = os.path.join(tmpdir, "projected")
            cluster_dir = os.path.join(tmpdir, "cluster")
            input_bundle_root = os.path.join(tmpdir, "bundle")
            output_root = os.path.join(tmpdir, "output")
            os.makedirs(projected_dir, exist_ok=True)
            os.makedirs(cluster_dir, exist_ok=True)
            os.makedirs(input_bundle_root, exist_ok=True)

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

            with open(os.path.join(input_bundle_root, "task_context.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "baseline_seed": 3,
                        "baseline_budget": 2,
                        "round_count": 3,
                    },
                    f,
                )

            for round_id in range(1, 4):
                with open(
                    os.path.join(input_bundle_root, f"recommendation_round_{round_id}.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        {
                            "round_id": round_id,
                            "baseline_id": f"baseline_round_{round_id}",
                            "hypothesis": {"summary": f"round {round_id} hypothesis"},
                            "controls": {
                                "direction": "explore",
                                "magnitude": "medium",
                                "complexity": "beam",
                                "risk": "balanced",
                                "slice_action_hints": [],
                            },
                            "candidate_rankings": [],
                            "search_tree": {"root_id": "root", "nodes": []},
                            "recommended_candidate_id": None,
                        },
                        f,
                    )

            baseline_manifest_path = os.path.join(tmpdir, "baseline_3.json")
            with open(baseline_manifest_path, "w", encoding="utf-8") as f:
                json.dump({"sample_ids": ["a.jpg", "b.jpg"]}, f)

            parser = build_parser()
            args = parser.parse_args(
                [
                    "--projected-dir",
                    projected_dir,
                    "--cluster-dir",
                    cluster_dir,
                    "--schema-path",
                    schema_path,
                    "--input-bundle-root",
                    input_bundle_root,
                    "--output-root",
                    output_root,
                    "--portrait-source",
                    "projected",
                    "--baseline-manifest-path",
                    baseline_manifest_path,
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
            self.assertTrue(os.path.exists(os.path.join(output_root, "prior_graph.json")))
            self.assertTrue(os.path.exists(os.path.join(output_root, "recommendation_round_1.json")))
            self.assertTrue(os.path.exists(os.path.join(output_root, "recommendation_round_2.json")))
            self.assertTrue(os.path.exists(os.path.join(output_root, "recommendation_round_3.json")))

            with open(os.path.join(output_root, "prior_graph.json"), "r", encoding="utf-8") as f:
                prior_graph = json.load(f)
            self.assertEqual(len(prior_graph["nodes"]), 3)

            with open(os.path.join(output_root, "recommendation_round_1.json"), "r", encoding="utf-8") as f:
                round_one = json.load(f)
            self.assertEqual(round_one["hypothesis"]["summary"], "round 1 hypothesis")
            self.assertTrue(round_one["candidate_rankings"])
            self.assertIn("search_tree", round_one)
            self.assertEqual(
                round_one["recommended_candidate_id"],
                round_one["candidate_rankings"][0]["candidate_id"],
            )
            self.assertIn("transfer_pairs", round_one["candidate_rankings"][0])
            self.assertEqual(len(round_one["candidate_rankings"][0]["delta_q"]), 3)


if __name__ == "__main__":
    unittest.main()
