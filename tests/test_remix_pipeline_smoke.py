import json
import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_remix_attach_results import main as attach_main
from clip_dinoiser.run_remix_analysis_report import main as analysis_main
from clip_dinoiser.run_remix_collect_results import main as collect_main
from clip_dinoiser.run_remix_recommendation import main as recommend_main
from clip_dinoiser.run_remix_response_dataset import build_parser as response_build_parser, main as response_main, run as response_run
from clip_dinoiser.run_remix_validate_recommendation import main as validate_recommendation_main


class RemixPipelineSmokeTests(unittest.TestCase):
    def test_response_dataset_run_emits_stage_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            projected_dir = os.path.join(tmpdir, "projected")
            cluster_dir = os.path.join(tmpdir, "cluster")
            os.makedirs(projected_dir, exist_ok=True)
            os.makedirs(cluster_dir, exist_ok=True)

            sample_ids = np.asarray(["a.jpg", "b.jpg", "c.jpg", "d.jpg"], dtype=object)
            np.savez(
                os.path.join(projected_dir, "projected_features.npz"),
                matrix=np.asarray([[0.1, 1.0], [0.2, 0.8], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32),
                sample_ids=sample_ids,
            )
            with open(os.path.join(projected_dir, "projected_features_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"sample_ids": sample_ids.tolist(), "block_ranges": {"quality.laplacian": [0, 2]}}, f)
            membership = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]], dtype=np.float32)
            np.savez(
                os.path.join(cluster_dir, "slice_result.npz"),
                sample_ids=sample_ids,
                membership=membership,
                hard_assignment=np.asarray([0, 0, 1, 1], dtype=np.int64),
                slice_weights=membership.mean(axis=0),
                centers=np.asarray([[0.15, 0.9], [0.85, 0.15]], dtype=np.float32),
            )
            with open(os.path.join(cluster_dir, "slice_result_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"finder": "gmm", "num_slices": 2}, f)

            logs = []
            args = response_build_parser().parse_args(
                [
                    "--projected-dir",
                    projected_dir,
                    "--cluster-dir",
                    cluster_dir,
                    "--output-path",
                    os.path.join(tmpdir, "rows.jsonl"),
                    "--budget",
                    "2",
                ]
            )

            self.assertEqual(response_run(args, log_fn=logs.append), 0)
            self.assertTrue(any("loading projected artifacts" in msg for msg in logs))
            self.assertTrue(any("loading slice artifacts" in msg for msg in logs))
            self.assertTrue(any("computing slice portraits" in msg for msg in logs))
            self.assertTrue(any("writing response rows" in msg for msg in logs))

    def test_response_dataset_rows_use_realized_sample_aggregation_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            projected_dir = os.path.join(tmpdir, "projected")
            cluster_dir = os.path.join(tmpdir, "cluster")
            manifest_dir = os.path.join(tmpdir, "manifests")
            os.makedirs(projected_dir, exist_ok=True)
            os.makedirs(cluster_dir, exist_ok=True)

            sample_ids = np.asarray(["a.jpg", "b.jpg", "c.jpg", "d.jpg"], dtype=object)
            projected_matrix = np.asarray(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [2.0, 2.0],
                    [4.0, 4.0],
                ],
                dtype=np.float32,
            )
            np.savez(
                os.path.join(projected_dir, "projected_features.npz"),
                matrix=projected_matrix,
                sample_ids=sample_ids,
            )
            with open(os.path.join(projected_dir, "projected_features_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"sample_ids": sample_ids.tolist(), "block_ranges": {"quality.laplacian": [0, 2]}}, f)

            membership = np.asarray(
                [
                    [0.95, 0.05],
                    [0.80, 0.20],
                    [0.20, 0.80],
                    [0.05, 0.95],
                ],
                dtype=np.float32,
            )
            np.savez(
                os.path.join(cluster_dir, "slice_result.npz"),
                sample_ids=sample_ids,
                membership=membership,
                hard_assignment=np.asarray([0, 0, 1, 1], dtype=np.int64),
                slice_weights=membership.mean(axis=0),
                centers=np.asarray([[0.5, 0.5], [3.0, 3.0]], dtype=np.float32),
            )
            with open(os.path.join(cluster_dir, "slice_result_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"finder": "gmm", "num_slices": 2}, f)

            rows_path = os.path.join(tmpdir, "rows.jsonl")
            self.assertEqual(
                response_main(
                    [
                        "--projected-dir",
                        projected_dir,
                        "--cluster-dir",
                        cluster_dir,
                        "--output-path",
                        rows_path,
                        "--budget",
                        "2",
                        "--subset-manifest-dir",
                        manifest_dir,
                    ]
                ),
                0,
            )

            with open(rows_path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            self.assertGreater(len(rows), 0)
            row = rows[0]
            self.assertEqual(row["feature_description_mode"], "realized_sample_aggregation")

            with open(row["execution"]["baseline_manifest_path"], "r", encoding="utf-8") as f:
                baseline_manifest = json.load(f)

            sample_index = {sample_id: index for index, sample_id in enumerate(sample_ids.tolist())}
            baseline_indices = [sample_index[sample_id] for sample_id in baseline_manifest["sample_ids"]]
            target_indices = [sample_index[sample_id] for sample_id in row["execution"]["selected_sample_ids"]]

            expected_baseline = projected_matrix[np.asarray(baseline_indices, dtype=np.int64)].mean(axis=0)
            expected_target = projected_matrix[np.asarray(target_indices, dtype=np.int64)].mean(axis=0)
            expected_delta = expected_target - expected_baseline

            self.assertTrue(
                np.allclose(
                    row["baseline_features_raw"]["quality.laplacian"],
                    expected_baseline.tolist(),
                )
            )
            self.assertTrue(
                np.allclose(
                    row["target_features_raw"]["quality.laplacian"],
                    expected_target.tolist(),
                )
            )
            self.assertTrue(
                np.allclose(
                    row["delta_phi"]["quality.laplacian"],
                    expected_delta.tolist(),
                )
            )

    def test_beam_v1_response_and_recommendation_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            projected_dir = os.path.join(tmpdir, "projected")
            cluster_dir = os.path.join(tmpdir, "cluster")
            os.makedirs(projected_dir, exist_ok=True)
            os.makedirs(cluster_dir, exist_ok=True)

            sample_ids = np.asarray(["a.jpg", "b.jpg", "c.jpg", "d.jpg"], dtype=object)
            projected_matrix = np.asarray(
                [
                    [0.1, 1.0, 0.2],
                    [0.2, 0.8, 0.1],
                    [0.9, 0.1, 0.7],
                    [0.8, 0.2, 0.6],
                ],
                dtype=np.float32,
            )
            np.savez(
                os.path.join(projected_dir, "projected_features.npz"),
                matrix=projected_matrix,
                sample_ids=sample_ids,
            )
            with open(os.path.join(projected_dir, "projected_features_meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "sample_ids": sample_ids.tolist(),
                        "block_ranges": {
                            "quality.laplacian": [0, 2],
                            "coverage.knn_local_density": [2, 3],
                        },
                    },
                    f,
                )

            membership = np.asarray(
                [
                    [0.9, 0.1],
                    [0.8, 0.2],
                    [0.2, 0.8],
                    [0.1, 0.9],
                ],
                dtype=np.float32,
            )
            np.savez(
                os.path.join(cluster_dir, "slice_result.npz"),
                sample_ids=sample_ids,
                membership=membership,
                hard_assignment=np.asarray([0, 0, 1, 1], dtype=np.int64),
                slice_weights=membership.mean(axis=0),
                centers=np.asarray([[0.15, 0.9, 0.15], [0.85, 0.15, 0.65]], dtype=np.float32),
            )
            with open(os.path.join(cluster_dir, "slice_result_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"finder": "gmm", "num_slices": 2}, f)

            rows_path = os.path.join(tmpdir, "rows.jsonl")
            self.assertEqual(
                response_main(
                    [
                        "--projected-dir",
                        projected_dir,
                        "--cluster-dir",
                        cluster_dir,
                        "--output-path",
                        rows_path,
                        "--budget",
                        "2",
                        "--pair-selector",
                        "beam_v1",
                    ]
                ),
                0,
            )

            with open(rows_path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            self.assertGreater(len(rows), 0)
            for index, row in enumerate(rows):
                row["measured_gain"] = 0.2 if index % 2 == 0 else -0.1

            labeled_rows_path = os.path.join(tmpdir, "rows_labeled.jsonl")
            with open(labeled_rows_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            output_path = os.path.join(tmpdir, "recommendation_beam.json")
            self.assertEqual(
                recommend_main(
                    [
                        "--projected-dir",
                        projected_dir,
                        "--cluster-dir",
                        cluster_dir,
                        "--response-dataset",
                        labeled_rows_path,
                        "--baseline-seed",
                        "0",
                        "--budget",
                        "2",
                        "--output-path",
                        output_path,
                        "--pair-selector",
                        "beam_v1",
                    ]
                ),
                0,
            )

            with open(output_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            self.assertIn("candidate_id", result)
            self.assertIn("ranked_candidates", result)
            self.assertGreater(len(result["ranked_candidates"]), 0)
            self.assertIn("search_tree", result)
            self.assertIsInstance(result["search_tree"], dict)
            self.assertIn("root_id", result["search_tree"])
            self.assertIn("nodes", result["search_tree"])
            self.assertGreater(len(result["search_tree"]["nodes"]), 0)
            completed_nodes = [
                node for node in result["search_tree"]["nodes"] if node.get("node_type") == "completed"
            ]
            self.assertGreater(len(completed_nodes), 0)
            ranked_candidate_ids = {candidate["candidate_id"] for candidate in result["ranked_candidates"]}
            completed_candidate_ids = {
                node.get("candidate_id") for node in completed_nodes if node.get("candidate_id")
            }
            self.assertTrue(completed_candidate_ids.issubset(ranked_candidate_ids))

    def test_remix_pipeline_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            projected_dir = os.path.join(tmpdir, "projected")
            cluster_dir = os.path.join(tmpdir, "cluster")
            os.makedirs(projected_dir, exist_ok=True)
            os.makedirs(cluster_dir, exist_ok=True)

            sample_ids = np.asarray(["a.jpg", "b.jpg", "c.jpg", "d.jpg"], dtype=object)
            projected_matrix = np.asarray(
                [
                    [0.1, 1.0, 0.2],
                    [0.2, 0.8, 0.1],
                    [0.9, 0.1, 0.7],
                    [0.8, 0.2, 0.6],
                ],
                dtype=np.float32,
            )
            np.savez(
                os.path.join(projected_dir, "projected_features.npz"),
                matrix=projected_matrix,
                sample_ids=sample_ids,
            )
            with open(os.path.join(projected_dir, "projected_features_meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "sample_ids": sample_ids.tolist(),
                        "block_ranges": {
                            "quality.laplacian": [0, 2],
                            "coverage.knn_local_density": [2, 3],
                        },
                    },
                    f,
                )

            membership = np.asarray(
                [
                    [0.9, 0.1],
                    [0.8, 0.2],
                    [0.2, 0.8],
                    [0.1, 0.9],
                ],
                dtype=np.float32,
            )
            np.savez(
                os.path.join(cluster_dir, "slice_result.npz"),
                sample_ids=sample_ids,
                membership=membership,
                hard_assignment=np.asarray([0, 0, 1, 1], dtype=np.int64),
                slice_weights=membership.mean(axis=0),
                centers=np.asarray([[0.15, 0.9, 0.15], [0.85, 0.15, 0.65]], dtype=np.float32),
            )
            with open(os.path.join(cluster_dir, "slice_result_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"finder": "gmm", "num_slices": 2}, f)

            rows_path = os.path.join(tmpdir, "rows.jsonl")
            manifest_dir = os.path.join(tmpdir, "subset_manifests")
            self.assertEqual(
                response_main(
                    [
                        "--projected-dir",
                        projected_dir,
                        "--cluster-dir",
                        cluster_dir,
                        "--output-path",
                        rows_path,
                        "--budget",
                        "2",
                        "--subset-manifest-dir",
                        manifest_dir,
                        "--pool-image-root",
                        tmpdir,
                    ]
                ),
                0,
            )

            with open(rows_path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            self.assertGreater(len(rows), 0)
            self.assertTrue(os.path.isdir(manifest_dir))
            self.assertGreater(len(os.listdir(manifest_dir)), 0)
            results_dir = os.path.join(tmpdir, "results")
            os.makedirs(results_dir, exist_ok=True)
            baseline_entry_written = set()
            for index, row in enumerate(rows):
                self.assertIn("baseline_manifest_path", row["execution"])
                baseline_manifest_path = row["execution"]["baseline_manifest_path"]
                baseline_id = os.path.splitext(os.path.basename(baseline_manifest_path))[0]
                if baseline_id not in baseline_entry_written:
                    baseline_result_path = os.path.join(results_dir, f"{baseline_id}.json")
                    with open(baseline_result_path, "w", encoding="utf-8") as baseline_f:
                        json.dump({"coco_stuff": {"summary": {"mIoU": 24.0}}}, baseline_f)
                    with open(os.path.join(results_dir, f"{baseline_id}_result_entry.json"), "w", encoding="utf-8") as entry_f:
                        json.dump(
                            {
                                "candidate_id": baseline_id,
                                "result_path": baseline_result_path,
                                "seed": 0,
                            },
                            entry_f,
                        )
                    baseline_entry_written.add(baseline_id)

                candidate_result_path = os.path.join(results_dir, f"{row['candidate_id']}.json")
                with open(candidate_result_path, "w", encoding="utf-8") as candidate_f:
                    json.dump(
                        {"coco_stuff": {"summary": {"mIoU": 24.2 if index % 2 == 0 else 23.9}}},
                        candidate_f,
                    )
                with open(os.path.join(results_dir, f"{row['candidate_id']}_result_entry.json"), "w", encoding="utf-8") as entry_f:
                    json.dump(
                        {
                            "candidate_id": row["candidate_id"],
                            "result_path": candidate_result_path,
                            "seed": 0,
                        },
                        entry_f,
                    )

            result_manifest_path = os.path.join(tmpdir, "result_manifest.jsonl")
            self.assertEqual(
                collect_main(
                    [
                        "--rows-path",
                        rows_path,
                        "--results-dir",
                        results_dir,
                        "--output-path",
                        result_manifest_path,
                    ]
                ),
                0,
            )

            labeled_rows_path = os.path.join(tmpdir, "rows_labeled.jsonl")
            self.assertEqual(
                attach_main(
                    [
                        "--rows-path",
                        rows_path,
                        "--result-manifest",
                        result_manifest_path,
                        "--metric-path",
                        "coco_stuff.summary.mIoU",
                        "--output-path",
                        labeled_rows_path,
                    ]
                ),
                0,
            )

            output_path = os.path.join(tmpdir, "recommendation.json")
            self.assertEqual(
                recommend_main(
                    [
                        "--projected-dir",
                        projected_dir,
                        "--cluster-dir",
                        cluster_dir,
                        "--response-dataset",
                        labeled_rows_path,
                        "--baseline-seed",
                        "0",
                        "--budget",
                        "2",
                        "--output-path",
                        output_path,
                    ]
                ),
                0,
            )

            with open(output_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            self.assertIn("candidate_id", result)
            self.assertIn("baseline_mixture", result)
            self.assertIn("target_mixture", result)
            self.assertIn("delta_q", result)
            self.assertIn("context", result)
            self.assertIn("portrait_summary", result)
            self.assertIn("ranked_candidates", result)
            self.assertGreater(len(result["ranked_candidates"]), 0)
            self.assertTrue(os.path.isfile(result["context"]["surrogate_output_path"]))

            recommendation_manifest_path = os.path.join(tmpdir, "recommended_manifest.json")
            self.assertEqual(
                validate_recommendation_main(
                    [
                        "--cluster-dir",
                        cluster_dir,
                        "--recommendation-path",
                        output_path,
                        "--pool-image-root",
                        tmpdir,
                        "--output-manifest",
                        recommendation_manifest_path,
                    ]
                ),
                0,
            )
            with open(recommendation_manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["candidate_id"], result["candidate_id"])
            self.assertEqual(len(manifest["sample_ids"]), 2)

            analysis_output_path = os.path.join(tmpdir, "analysis_report.json")
            self.assertEqual(
                analysis_main(
                    [
                        "--response-dataset",
                        labeled_rows_path,
                        "--recommendation-path",
                        output_path,
                        "--output-path",
                        analysis_output_path,
                    ]
                ),
                0,
            )
            with open(analysis_output_path, "r", encoding="utf-8") as f:
                analysis_report = json.load(f)
            self.assertIn("surrogate", analysis_report)
            self.assertIn("cross_validation", analysis_report["surrogate"])
            self.assertIn("actual_comparison", analysis_report["recommendation"])


if __name__ == "__main__":
    unittest.main()
