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


from clip_dinoiser.run_slice_finding_baseline import main
from tests.test_processed_feature_assembler import TEST_SCHEMA


class SliceBaselineCliTests(unittest.TestCase):
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

    def test_cli_writes_membership_and_metadata_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root, schema_path = self._write_fixture_bundle(tmpdir)
            output_dir = os.path.join(tmpdir, "artifacts")

            exit_code = main(
                [
                    "--data-root",
                    data_root,
                    "--schema-path",
                    schema_path,
                    "--output-dir",
                    output_dir,
                    "--finder",
                    "soft_kmeans",
                    "--num-slices",
                    "2",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "slice_result.npz")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "slice_result_meta.json")))

            payload = np.load(os.path.join(output_dir, "slice_result.npz"), allow_pickle=True)
            self.assertEqual(payload["membership"].shape, (3, 2))
            self.assertEqual(payload["centers"].shape, (2, 12))

            with open(os.path.join(output_dir, "slice_result_meta.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["finder"], "soft_kmeans")
            self.assertEqual(meta["num_slices"], 2)
            self.assertEqual(meta["sample_count"], 3)

    def test_script_file_can_run_help_without_package_context(self):
        script_path = os.path.join(ROOT, "clip_dinoiser", "run_slice_finding_baseline.py")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)

        result = subprocess.run(
            [sys.executable, script_path, "--help"],
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Run the slice discovery baseline pipeline", result.stdout)


if __name__ == "__main__":
    unittest.main()
