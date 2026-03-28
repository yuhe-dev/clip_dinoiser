import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_slice_assembler_debug import main as assembler_main
from clip_dinoiser.run_slice_assembler_probe import main as assembler_probe_main
from clip_dinoiser.run_slice_bundle_probe import main as probe_main
from clip_dinoiser.run_slice_cluster_debug import main as cluster_main
from clip_dinoiser.run_slice_projector_debug import main as projector_main
from tests.test_processed_feature_assembler import TEST_SCHEMA


class SliceDebugScriptTests(unittest.TestCase):
    def _write_fixture_bundle(self, root_dir: str) -> tuple[str, str]:
        data_root = os.path.join(root_dir, "data_feature")
        os.makedirs(os.path.join(data_root, "quality"), exist_ok=True)
        os.makedirs(os.path.join(data_root, "difficulty"), exist_ok=True)
        os.makedirs(os.path.join(data_root, "coverage"), exist_ok=True)

        quality_records = [
            {
                "image_rel": "images/train2017/0001.jpg",
                "annotation_rel": "annotations/train2017/0001_labelTrainIds.png",
                "schema_version": "quality.v1",
                "features": {
                    "laplacian": {
                        "hist": np.asarray([0.1, 0.2], dtype=np.float32),
                        "log_num_values": 1.0,
                        "empty_flag": 0,
                    }
                },
            },
            {
                "image_rel": "images/train2017/0002.jpg",
                "annotation_rel": "annotations/train2017/0002_labelTrainIds.png",
                "schema_version": "quality.v1",
                "features": {
                    "laplacian": {
                        "hist": np.asarray([0.3, 0.4], dtype=np.float32),
                        "log_num_values": 2.0,
                        "empty_flag": 0,
                    }
                },
            },
            {
                "image_rel": "images/train2017/0003.jpg",
                "annotation_rel": "annotations/train2017/0003_labelTrainIds.png",
                "schema_version": "quality.v1",
                "features": {
                    "laplacian": {
                        "hist": np.asarray([0.7, 0.8], dtype=np.float32),
                        "log_num_values": 3.0,
                        "empty_flag": 0,
                    }
                },
            },
        ]
        difficulty_records = [
            {
                "image_rel": "images/train2017/0001.jpg",
                "annotation_rel": "annotations/train2017/0001_labelTrainIds.png",
                "schema_version": "difficulty.v1",
                "features": {
                    "small_ratio": {
                        "delta_profile": np.asarray([0.2, 0.3], dtype=np.float32),
                        "log_num_values": 1.0,
                        "empty_flag": 0,
                        "summary": {"mass_small_extreme": 0.5},
                    }
                },
            },
            {
                "image_rel": "images/train2017/0002.jpg",
                "annotation_rel": "annotations/train2017/0002_labelTrainIds.png",
                "schema_version": "difficulty.v1",
                "features": {
                    "small_ratio": {
                        "delta_profile": np.asarray([0.4, 0.1], dtype=np.float32),
                        "log_num_values": 1.5,
                        "empty_flag": 0,
                        "summary": {"mass_small_extreme": 0.7},
                    }
                },
            },
            {
                "image_rel": "images/train2017/0003.jpg",
                "annotation_rel": "annotations/train2017/0003_labelTrainIds.png",
                "schema_version": "difficulty.v1",
                "features": {
                    "small_ratio": {
                        "delta_profile": np.asarray([0.8, 0.2], dtype=np.float32),
                        "log_num_values": 2.0,
                        "empty_flag": 0,
                        "summary": {"mass_small_extreme": 0.9},
                    }
                },
            },
        ]
        coverage_records = [
            {
                "image_rel": "images/train2017/0001.jpg",
                "annotation_rel": "annotations/train2017/0001_labelTrainIds.png",
                "schema_version": "coverage.v1",
                "features": {
                    "knn_local_density": {
                        "profile": np.asarray([0.6, 0.8], dtype=np.float32),
                        "summary": {"q50": 0.7},
                    }
                },
            },
            {
                "image_rel": "images/train2017/0002.jpg",
                "annotation_rel": "annotations/train2017/0002_labelTrainIds.png",
                "schema_version": "coverage.v1",
                "features": {
                    "knn_local_density": {
                        "profile": np.asarray([0.5, 0.9], dtype=np.float32),
                        "summary": {"q50": 0.75},
                    }
                },
            },
            {
                "image_rel": "images/train2017/0003.jpg",
                "annotation_rel": "annotations/train2017/0003_labelTrainIds.png",
                "schema_version": "coverage.v1",
                "features": {
                    "knn_local_density": {
                        "profile": np.asarray([0.9, 1.0], dtype=np.float32),
                        "summary": {"q50": 0.95},
                    }
                },
            },
        ]

        np.save(os.path.join(data_root, "quality", "quality_processed_features.npy"), np.asarray(quality_records, dtype=object), allow_pickle=True)
        np.save(os.path.join(data_root, "difficulty", "difficulty_processed_features.npy"), np.asarray(difficulty_records, dtype=object), allow_pickle=True)
        np.save(os.path.join(data_root, "coverage", "coverage_processed_features.npy"), np.asarray(coverage_records, dtype=object), allow_pickle=True)

        schema_path = os.path.join(root_dir, "schema.json")
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(TEST_SCHEMA, f, indent=2, ensure_ascii=False)

        return data_root, schema_path

    def test_assembler_debug_script_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root, schema_path = self._write_fixture_bundle(tmpdir)
            output_dir = os.path.join(tmpdir, "assembled")

            exit_code = assembler_main(
                [
                    "--data-root",
                    data_root,
                    "--schema-path",
                    schema_path,
                    "--output-dir",
                    output_dir,
                    "--limit-samples",
                    "2",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "assembled_features.npz")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "assembled_features_meta.json")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "assembler_debug.json")))

    def test_bundle_probe_script_reports_stage_statuses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root, schema_path = self._write_fixture_bundle(tmpdir)
            output_path = os.path.join(tmpdir, "bundle_probe.json")

            exit_code = probe_main(
                [
                    "--data-root",
                    data_root,
                    "--schema-path",
                    schema_path,
                    "--output-path",
                    output_path,
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(output_path))

            with open(output_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["stages"]["schema"]["status"], "ok")
            self.assertEqual(payload["stages"]["quality"]["status"], "ok")
            self.assertEqual(payload["stages"]["difficulty"]["status"], "ok")
            self.assertEqual(payload["stages"]["coverage"]["status"], "ok")

    def test_assembler_probe_script_reports_substage_statuses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root, schema_path = self._write_fixture_bundle(tmpdir)
            output_path = os.path.join(tmpdir, "assembler_probe.json")

            exit_code = assembler_probe_main(
                [
                    "--data-root",
                    data_root,
                    "--schema-path",
                    schema_path,
                    "--output-path",
                    output_path,
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(output_path))

            with open(output_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["stages"]["load_quality"]["status"], "ok")
            self.assertEqual(payload["stages"]["load_difficulty"]["status"], "ok")
            self.assertEqual(payload["stages"]["load_coverage"]["status"], "ok")
            self.assertEqual(payload["stages"]["validate_alignment"]["status"], "ok")
            self.assertEqual(payload["stages"]["build_quality_blocks"]["status"], "ok")
            self.assertEqual(payload["stages"]["build_difficulty_blocks"]["status"], "ok")
            self.assertEqual(payload["stages"]["build_coverage_blocks"]["status"], "ok")
            self.assertEqual(payload["stages"]["flat_view"]["status"], "ok")
            self.assertEqual(payload["stages"]["save"]["status"], "ok")

    def test_projector_debug_script_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root, schema_path = self._write_fixture_bundle(tmpdir)
            assembler_dir = os.path.join(tmpdir, "assembled")
            projector_dir = os.path.join(tmpdir, "projected")

            assembler_main(
                [
                    "--data-root",
                    data_root,
                    "--schema-path",
                    schema_path,
                    "--output-dir",
                    assembler_dir,
                ]
            )

            exit_code = projector_main(
                [
                    "--assembled-dir",
                    assembler_dir,
                    "--output-dir",
                    projector_dir,
                    "--scalar-scaler",
                    "zscore",
                    "--block-weighting",
                    "equal_by_block",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(os.path.join(projector_dir, "projected_features.npz")))
            self.assertTrue(os.path.exists(os.path.join(projector_dir, "projected_features_meta.json")))
            self.assertTrue(os.path.exists(os.path.join(projector_dir, "projector_debug.json")))

    def test_cluster_debug_script_writes_artifacts_for_soft_kmeans_gmm_and_vmf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root, schema_path = self._write_fixture_bundle(tmpdir)
            assembler_dir = os.path.join(tmpdir, "assembled")
            projector_dir = os.path.join(tmpdir, "projected")
            soft_dir = os.path.join(tmpdir, "soft")
            gmm_dir = os.path.join(tmpdir, "gmm")
            vmf_dir = os.path.join(tmpdir, "vmf")

            assembler_main(
                [
                    "--data-root",
                    data_root,
                    "--schema-path",
                    schema_path,
                    "--output-dir",
                    assembler_dir,
                ]
            )
            projector_main(
                [
                    "--assembled-dir",
                    assembler_dir,
                    "--output-dir",
                    projector_dir,
                ]
            )

            soft_exit = cluster_main(
                [
                    "--projected-dir",
                    projector_dir,
                    "--output-dir",
                    soft_dir,
                    "--finder",
                    "soft_kmeans",
                    "--num-slices",
                    "2",
                ]
            )
            gmm_exit = cluster_main(
                [
                    "--projected-dir",
                    projector_dir,
                    "--output-dir",
                    gmm_dir,
                    "--finder",
                    "gmm",
                    "--num-slices",
                    "2",
                ]
            )
            vmf_exit = cluster_main(
                [
                    "--projected-dir",
                    projector_dir,
                    "--output-dir",
                    vmf_dir,
                    "--finder",
                    "vmf",
                    "--num-slices",
                    "2",
                ]
            )

            self.assertEqual(soft_exit, 0)
            self.assertEqual(gmm_exit, 0)
            self.assertEqual(vmf_exit, 0)
            self.assertTrue(os.path.exists(os.path.join(soft_dir, "cluster_debug.json")))
            self.assertTrue(os.path.exists(os.path.join(gmm_dir, "cluster_debug.json")))
            self.assertTrue(os.path.exists(os.path.join(vmf_dir, "cluster_debug.json")))

            with open(os.path.join(gmm_dir, "cluster_debug.json"), "r", encoding="utf-8") as f:
                gmm_debug = json.load(f)
            self.assertEqual(gmm_debug["finder"], "gmm")
            self.assertIn("log_likelihood_trace", gmm_debug)

            with open(os.path.join(vmf_dir, "cluster_debug.json"), "r", encoding="utf-8") as f:
                vmf_debug = json.load(f)
            self.assertEqual(vmf_debug["finder"], "vmf")
            self.assertIn("mean_kappa_trace", vmf_debug)

    def test_debug_scripts_can_run_help_without_package_context(self):
        script_names = [
            "run_slice_bundle_probe.py",
            "run_slice_assembler_debug.py",
            "run_slice_projector_debug.py",
            "run_slice_cluster_debug.py",
        ]
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)

        for script_name in script_names:
            script_path = os.path.join(ROOT, "clip_dinoiser", script_name)
            result = subprocess.run(
                [sys.executable, script_path, "--help"],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
