import json
import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_workbench_bundle_export import build_parser, run


class WorkbenchBundleExportCliTests(unittest.TestCase):
    def test_run_exports_real_bundle_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            projected_dir = os.path.join(tmpdir, "projected")
            cluster_dir = os.path.join(tmpdir, "cluster")
            slice_report_dir = os.path.join(tmpdir, "slice_report")
            input_bundle_root = os.path.join(tmpdir, "bundle")
            output_root = os.path.join(tmpdir, "output")
            os.makedirs(projected_dir, exist_ok=True)
            os.makedirs(cluster_dir, exist_ok=True)
            os.makedirs(slice_report_dir, exist_ok=True)
            os.makedirs(input_bundle_root, exist_ok=True)

            np.savez(
                os.path.join(projected_dir, "projected_features.npz"),
                matrix=np.asarray(
                    [
                        [0.7, 0.3, 0.8, 0.2],
                        [0.2, 0.8, 0.4, 0.9],
                        [0.6, 0.4, 0.7, 0.3],
                    ],
                    dtype=np.float32,
                ),
                sample_ids=np.asarray(["a.jpg", "b.jpg", "c.jpg"], dtype=object),
            )
            with open(os.path.join(projected_dir, "projected_features_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"block_ranges": {"quality.laplacian": [0, 3], "coverage.knn_local_density": [3, 4]}}, f)

            np.savez(
                os.path.join(cluster_dir, "slice_result.npz"),
                sample_ids=np.asarray(["a.jpg", "b.jpg", "c.jpg"], dtype=object),
                membership=np.asarray(
                    [
                        [0.9, 0.1],
                        [0.2, 0.8],
                        [0.7, 0.3],
                    ],
                    dtype=np.float32,
                ),
                hard_assignment=np.asarray([0, 1, 0], dtype=np.int64),
                slice_weights=np.asarray([0.6, 0.4], dtype=np.float32),
                centers=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            )
            with open(os.path.join(cluster_dir, "slice_result_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"finder": "gmm", "num_slices": 2}, f)

            schema = {
                "schema_version": "processed_feature_bundle.v1",
                "dimensions": {
                    "quality": {
                        "features": {
                            "laplacian": {
                                "model_input_fields": ["hist", "q50"],
                            }
                        }
                    },
                    "difficulty": {"features": {}},
                    "coverage": {
                        "features": {
                            "knn_local_density": {
                                "model_input_fields": ["density_score"],
                            }
                        }
                    },
                },
            }
            schema_path = os.path.join(tmpdir, "schema.json")
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(schema, f)

            with open(os.path.join(input_bundle_root, "task_context.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "baseline_seed": 0,
                        "baseline_budget": 2,
                        "round_count": 1,
                    },
                    f,
                )
            with open(os.path.join(input_bundle_root, "recommendation_round_1.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "candidate_rankings": [
                            {
                                "candidate_id": "cand_r1_1",
                                "delta_q": [-0.1, 0.1],
                            }
                        ]
                    },
                    f,
                )

            with open(os.path.join(slice_report_dir, "slices.json"), "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "slice_id": "slice_00",
                            "index": 0,
                            "top_shifted_features": [{"block": "quality.laplacian"}],
                            "representative_samples": ["a.jpg"],
                            "ambiguous_samples": ["c.jpg"],
                        },
                        {
                            "slice_id": "slice_01",
                            "index": 1,
                            "top_shifted_features": [{"block": "coverage.knn_local_density"}],
                            "representative_samples": ["b.jpg"],
                            "ambiguous_samples": ["c.jpg"],
                        },
                    ],
                    f,
                )
            with open(os.path.join(slice_report_dir, "samples.json"), "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "sample_id": "a.jpg",
                            "image_url": "./thumbnails/a.jpg",
                            "hard_assignment": 0,
                            "membership_vector": [0.9, 0.1],
                            "slice_rankings": [0, 1],
                        },
                        {
                            "sample_id": "b.jpg",
                            "image_url": "./thumbnails/b.jpg",
                            "hard_assignment": 1,
                            "membership_vector": [0.2, 0.8],
                            "slice_rankings": [1, 0],
                        },
                        {
                            "sample_id": "c.jpg",
                            "image_url": "./thumbnails/c.jpg",
                            "hard_assignment": 0,
                            "membership_vector": [0.7, 0.3],
                            "slice_rankings": [0, 1],
                        },
                    ],
                    f,
                )
            with open(os.path.join(slice_report_dir, "slice_centers_2d.json"), "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {"slice_id": "slice_00", "x": -1.0, "y": 0.0},
                        {"slice_id": "slice_01", "x": 1.0, "y": 0.0},
                    ],
                    f,
                )

            baseline_manifest_path = os.path.join(tmpdir, "baseline_0.json")
            with open(baseline_manifest_path, "w", encoding="utf-8") as f:
                json.dump({"sample_ids": ["a.jpg", "b.jpg"]}, f)
            baseline_result_path = os.path.join(tmpdir, "baseline_result.json")
            with open(baseline_result_path, "w", encoding="utf-8") as f:
                json.dump({"coco_stuff": {"summary": {"mIoU": 24.5}}}, f)

            parser = build_parser()
            args = parser.parse_args(
                [
                    "--projected-dir",
                    projected_dir,
                    "--cluster-dir",
                    cluster_dir,
                    "--slice-report-dir",
                    slice_report_dir,
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
                    "--baseline-result-path",
                    baseline_result_path,
                ]
            )

            status = run(args, log_fn=lambda _: None)

            self.assertEqual(status, 0)
            for name in [
                "prior_graph.json",
                "slice_atlas.json",
                "baseline_footprint.json",
                "field_distributions.json",
                "samples.json",
                "slice_relationships.json",
            ]:
                self.assertTrue(os.path.exists(os.path.join(output_root, name)))

            with open(os.path.join(output_root, "field_distributions.json"), "r", encoding="utf-8") as f:
                field_payload = json.load(f)
            quality_block = next(block for block in field_payload["blocks"] if block["block_name"] == "quality")
            coverage_block = next(block for block in field_payload["blocks"] if block["block_name"] == "coverage")
            histogram_field = quality_block["features"][0]["fields"][0]
            scalar_field = coverage_block["features"][0]["fields"][0]
            self.assertEqual(histogram_field["distribution_type"], "histogram")
            self.assertEqual(len(histogram_field["global_pool"]["values"]), 2)
            self.assertIn("cand_r1_1", histogram_field["candidate_expected"])
            self.assertEqual(scalar_field["distribution_type"], "scalar_interval")
            self.assertIn("summary", scalar_field["baseline"])

            with open(os.path.join(output_root, "baseline_footprint.json"), "r", encoding="utf-8") as f:
                footprint = json.load(f)
            self.assertEqual(footprint["baseline_performance"]["value"], 24.5)

            with open(os.path.join(output_root, "samples.json"), "r", encoding="utf-8") as f:
                samples_payload = json.load(f)
            self.assertTrue(samples_payload["samples"])
            self.assertEqual(samples_payload["samples"][0]["image_url"], "./thumbnails/a.jpg")


if __name__ == "__main__":
    unittest.main()
