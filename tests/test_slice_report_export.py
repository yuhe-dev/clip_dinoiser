import json
import os
import sys
import tempfile
import unittest

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

            with open(os.path.join(report_dir, "run_summary.json"), "r", encoding="utf-8") as f:
                run_summary = json.load(f)
            self.assertEqual(run_summary["finder"], "gmm")
            self.assertEqual(run_summary["num_slices"], 2)
            self.assertEqual(run_summary["sample_count"], 3)

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


if __name__ == "__main__":
    unittest.main()
