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
from clip_dinoiser.run_remix_collect_results import main as collect_main
from clip_dinoiser.run_remix_recommendation import main as recommend_main
from clip_dinoiser.run_remix_response_dataset import build_parser as response_build_parser, main as response_main, run as response_run


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
            self.assertIn("baseline_mixture", result)
            self.assertIn("target_mixture", result)
            self.assertIn("delta_q", result)


if __name__ == "__main__":
    unittest.main()
