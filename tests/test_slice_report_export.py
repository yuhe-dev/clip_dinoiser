import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_slice_cluster_debug import main as cluster_main
from clip_dinoiser.run_slice_report_export import main as export_main
from tests.test_slice_debug_scripts import SliceDebugScriptTests


class SliceReportExportTests(unittest.TestCase):
    def _write_dummy_images(self, root_dir: str) -> str:
        image_root = os.path.join(root_dir, "images")
        os.makedirs(os.path.join(image_root, "images", "train2017"), exist_ok=True)
        for image_name in ["0001.jpg", "0002.jpg", "0003.jpg"]:
            with open(os.path.join(image_root, "images", "train2017", image_name), "wb") as f:
                f.write(b"fake-image-bytes")
        return image_root

    def test_report_export_writes_core_json_artifacts(self):
        fixture_builder = SliceDebugScriptTests()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_root, schema_path = fixture_builder._write_fixture_bundle(tmpdir)
            assembled_dir = os.path.join(tmpdir, "assembled")
            projected_dir = os.path.join(tmpdir, "projected")
            cluster_dir = os.path.join(tmpdir, "cluster")
            report_dir = os.path.join(tmpdir, "report")

            from clip_dinoiser.run_slice_assembler_debug import main as assembler_main
            from clip_dinoiser.run_slice_projector_debug import main as projector_main

            assembler_main(
                [
                    "--data-root",
                    data_root,
                    "--schema-path",
                    schema_path,
                    "--output-dir",
                    assembled_dir,
                ]
            )
            projector_main(
                [
                    "--assembled-dir",
                    assembled_dir,
                    "--output-dir",
                    projected_dir,
                    "--scalar-scaler",
                    "zscore",
                    "--block-weighting",
                    "equal_by_block",
                ]
            )
            cluster_main(
                [
                    "--projected-dir",
                    projected_dir,
                    "--output-dir",
                    cluster_dir,
                    "--finder",
                    "gmm",
                    "--num-slices",
                    "2",
                ]
            )
            image_root = self._write_dummy_images(tmpdir)

            embedding = np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.5],
                    [0.5, 1.0],
                ],
                dtype=np.float32,
            )

            with patch(
                "clip_dinoiser.slice_discovery.report_exporter.SliceReportExporter._compute_umap_2d",
                return_value=embedding,
            ):
                exit_code = export_main(
                    [
                        "--projected-dir",
                        projected_dir,
                        "--cluster-dir",
                        cluster_dir,
                        "--output-dir",
                        report_dir,
                        "--image-root",
                        image_root,
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(os.path.join(report_dir, "run_summary.json")))
            self.assertTrue(os.path.exists(os.path.join(report_dir, "slices.json")))
            self.assertTrue(os.path.exists(os.path.join(report_dir, "samples.json")))
            self.assertTrue(os.path.exists(os.path.join(report_dir, "feature_schema.json")))
            self.assertTrue(os.path.exists(os.path.join(report_dir, "embedding_2d.json")))
            self.assertTrue(os.path.exists(os.path.join(report_dir, "slice_centers_2d.json")))

            with open(os.path.join(report_dir, "run_summary.json"), "r", encoding="utf-8") as f:
                run_summary = json.load(f)
            self.assertEqual(run_summary["finder"], "gmm")
            self.assertEqual(run_summary["num_slices"], 2)
            self.assertEqual(run_summary["sample_count"], 3)
            self.assertEqual(run_summary["embedding"]["method"], "umap")
            self.assertIn("random_state", run_summary["embedding"])
            self.assertIn("n_neighbors", run_summary["embedding"])
            self.assertIn("min_dist", run_summary["embedding"])

            with open(os.path.join(report_dir, "slices.json"), "r", encoding="utf-8") as f:
                slices = json.load(f)
            self.assertEqual(len(slices), 2)
            self.assertIn("slice_id", slices[0])
            self.assertIn("weight", slices[0])
            self.assertIn("hard_count", slices[0])
            self.assertIn("avg_entropy", slices[0])
            self.assertIn("block_portrait", slices[0])
            self.assertIn("top_shifted_features", slices[0])
            self.assertIn("representative_samples", slices[0])
            self.assertIn("center_samples", slices[0])
            self.assertIn("ambiguous_samples", slices[0])
            self.assertEqual(len(slices[0]["block_portrait"]), 3)

            with open(os.path.join(report_dir, "samples.json"), "r", encoding="utf-8") as f:
                samples = json.load(f)
            self.assertEqual(len(samples), 3)
            self.assertIn("sample_id", samples[0])
            self.assertIn("hard_assignment", samples[0])
            self.assertIn("membership_vector", samples[0])
            self.assertIn("image_url", samples[0])
            self.assertTrue(samples[0]["image_url"].startswith("./thumbnails/"))

            with open(os.path.join(report_dir, "feature_schema.json"), "r", encoding="utf-8") as f:
                feature_schema = json.load(f)
            self.assertEqual(feature_schema["block_order"][0], "quality.laplacian")
            self.assertIn("quality.laplacian", feature_schema["blocks"])
            self.assertTrue(os.path.exists(os.path.join(report_dir, "thumbnails")))

            with open(os.path.join(report_dir, "embedding_2d.json"), "r", encoding="utf-8") as f:
                embedding_payload = json.load(f)
            self.assertEqual(len(embedding_payload), 3)
            self.assertEqual(
                set(embedding_payload[0].keys()),
                {"sample_id", "x", "y", "hard_assignment", "max_membership", "display"},
            )
            self.assertTrue(np.isfinite([row["x"] for row in embedding_payload]).all())
            self.assertTrue(np.isfinite([row["y"] for row in embedding_payload]).all())

            with open(os.path.join(report_dir, "slice_centers_2d.json"), "r", encoding="utf-8") as f:
                center_payload = json.load(f)
            self.assertEqual(len(center_payload), 2)
            self.assertEqual(
                set(center_payload[0].keys()),
                {"slice_id", "x", "y", "weight", "hard_count"},
            )

    def test_report_export_only_copies_images_referenced_by_slice_views(self):
        fixture_builder = SliceDebugScriptTests()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_root, schema_path = fixture_builder._write_fixture_bundle(tmpdir)
            assembled_dir = os.path.join(tmpdir, "assembled")
            projected_dir = os.path.join(tmpdir, "projected")
            cluster_dir = os.path.join(tmpdir, "cluster")
            report_dir = os.path.join(tmpdir, "report")

            from clip_dinoiser.run_slice_assembler_debug import main as assembler_main
            from clip_dinoiser.run_slice_projector_debug import main as projector_main

            assembler_main(
                [
                    "--data-root",
                    data_root,
                    "--schema-path",
                    schema_path,
                    "--output-dir",
                    assembled_dir,
                ]
            )
            projector_main(
                [
                    "--assembled-dir",
                    assembled_dir,
                    "--output-dir",
                    projected_dir,
                    "--scalar-scaler",
                    "zscore",
                    "--block-weighting",
                    "equal_by_block",
                ]
            )
            cluster_main(
                [
                    "--projected-dir",
                    projected_dir,
                    "--output-dir",
                    cluster_dir,
                    "--finder",
                    "gmm",
                    "--num-slices",
                    "2",
                ]
            )
            image_root = self._write_dummy_images(tmpdir)

            def _only_first_sample(sample_ids, scores, descending, top_k=12):
                del scores, descending, top_k
                return [sample_ids[0]]

            embedding = np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.5],
                    [0.5, 1.0],
                ],
                dtype=np.float32,
            )

            with patch(
                "clip_dinoiser.slice_discovery.report_exporter.SliceReportExporter._top_sample_ids",
                side_effect=_only_first_sample,
            ), patch(
                "clip_dinoiser.slice_discovery.report_exporter.SliceReportExporter._compute_umap_2d",
                return_value=embedding,
            ), patch(
                "clip_dinoiser.slice_discovery.report_exporter.SliceReportExporter._select_display_sample_ids",
                return_value={"images/train2017/0002.jpg"},
            ):
                exit_code = export_main(
                    [
                        "--projected-dir",
                        projected_dir,
                        "--cluster-dir",
                        cluster_dir,
                        "--output-dir",
                        report_dir,
                        "--image-root",
                        image_root,
                    ]
                )

            self.assertEqual(exit_code, 0)

            thumbnail_files = []
            for root, _, files in os.walk(os.path.join(report_dir, "thumbnails")):
                for name in files:
                    thumbnail_files.append(os.path.relpath(os.path.join(root, name), os.path.join(report_dir, "thumbnails")))

            self.assertEqual(sorted(thumbnail_files), ["images/train2017/0001.jpg", "images/train2017/0002.jpg"])

            with open(os.path.join(report_dir, "samples.json"), "r", encoding="utf-8") as f:
                samples = json.load(f)

            non_empty_urls = [sample["image_url"] for sample in samples if sample["image_url"]]
            self.assertEqual(
                non_empty_urls,
                [
                    "./thumbnails/images/train2017/0001.jpg",
                    "./thumbnails/images/train2017/0002.jpg",
                ],
            )


if __name__ == "__main__":
    unittest.main()
