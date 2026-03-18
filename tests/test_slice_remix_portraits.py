import os
import sys
import json
import tempfile
import unittest
import subprocess

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_discovery.types import ProjectedSliceFeatures
from clip_dinoiser.slice_discovery.assembler import ProcessedFeatureAssembler
from clip_dinoiser.slice_remix.portraits import (
    build_feature_label_map,
    compute_portrait_shift,
    compute_slice_portraits,
    load_portrait_feature_groups,
    summarize_portrait_shift,
)
from tests.test_processed_feature_assembler import ProcessedFeatureAssemblerTests, TEST_SCHEMA


class SliceRemixPortraitTests(unittest.TestCase):
    def test_compute_portrait_shift_from_slice_portraits(self):
        feature_groups = {
            "feature_a": np.asarray([[1.0], [3.0], [5.0]], dtype=np.float32),
        }
        memberships = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        )

        portraits = compute_slice_portraits(feature_groups, memberships)
        baseline = np.asarray([0.5, 0.5], dtype=np.float32)
        target = np.asarray([0.7, 0.3], dtype=np.float32)
        shift = compute_portrait_shift(portraits, baseline, target)

        self.assertIn("feature_a", shift)
        self.assertEqual(shift["feature_a"].shape, (1,))

    def test_load_portrait_feature_groups_prefers_semantic_processed_blocks(self):
        fixture = ProcessedFeatureAssemblerTests()
        quality_records, difficulty_records, coverage_records = fixture._build_records()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = os.path.join(tmpdir, "data_feature")
            os.makedirs(os.path.join(data_root, "quality"), exist_ok=True)
            os.makedirs(os.path.join(data_root, "difficulty"), exist_ok=True)
            os.makedirs(os.path.join(data_root, "coverage"), exist_ok=True)
            np.save(
                os.path.join(data_root, "quality", "quality_processed_features.npy"),
                np.asarray(quality_records, dtype=object),
                allow_pickle=True,
            )
            np.save(
                os.path.join(data_root, "difficulty", "difficulty_processed_features.npy"),
                np.asarray(difficulty_records, dtype=object),
                allow_pickle=True,
            )
            np.save(
                os.path.join(data_root, "coverage", "coverage_processed_features.npy"),
                np.asarray(coverage_records, dtype=object),
                allow_pickle=True,
            )
            schema_path = os.path.join(tmpdir, "schema.json")
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(TEST_SCHEMA, f, indent=2, ensure_ascii=False)

            projected = ProjectedSliceFeatures(
                matrix=np.zeros((2, 12), dtype=np.float32),
                sample_ids=["images/train2017/0001.jpg", "images/train2017/0002.jpg"],
                block_ranges={
                    "quality.laplacian": (0, 4),
                    "difficulty.small_ratio": (4, 9),
                    "coverage.knn_local_density": (9, 12),
                },
            )

            groups, source = load_portrait_feature_groups(
                projected=projected,
                cluster_meta={
                    "data_root": data_root,
                    "schema_path": schema_path,
                },
                portrait_source="auto",
            )

        self.assertEqual(source, "semantic")
        self.assertEqual(sorted(groups.keys()), sorted(projected.block_ranges.keys()))
        np.testing.assert_allclose(
            groups["quality.laplacian"][0],
            np.asarray([0.1, 0.2, 1.0, 0.0], dtype=np.float32),
        )

    def test_load_portrait_feature_groups_prefers_assembled_feature_cache(self):
        fixture = ProcessedFeatureAssemblerTests()
        quality_records, difficulty_records, coverage_records = fixture._build_records()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = os.path.join(tmpdir, "data_feature")
            os.makedirs(data_root, exist_ok=True)
            schema_path = os.path.join(tmpdir, "schema.json")
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(TEST_SCHEMA, f, indent=2, ensure_ascii=False)

            assembler = ProcessedFeatureAssembler.from_processed_records(
                quality_records=quality_records,
                difficulty_records=difficulty_records,
                coverage_records=coverage_records,
                schema=TEST_SCHEMA,
            )
            assembled_dir = os.path.join(data_root, "assembled_features")
            assembler.save(assembled_dir)

            projected = ProjectedSliceFeatures(
                matrix=np.zeros((2, 12), dtype=np.float32),
                sample_ids=["images/train2017/0001.jpg", "images/train2017/0002.jpg"],
                block_ranges={
                    "quality.laplacian": (0, 4),
                    "difficulty.small_ratio": (4, 9),
                    "coverage.knn_local_density": (9, 12),
                },
            )

            logs: list[str] = []
            groups, source = load_portrait_feature_groups(
                projected=projected,
                cluster_meta={
                    "data_root": data_root,
                    "schema_path": schema_path,
                },
                portrait_source="auto",
                log_fn=logs.append,
            )

        self.assertEqual(source, "semantic")
        self.assertTrue(any("loading semantic assembled features" in message for message in logs))
        self.assertEqual(sorted(groups.keys()), sorted(projected.block_ranges.keys()))

    def test_run_remix_response_dataset_module_is_importable_in_script_mode(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os, sys; sys.path.insert(0, os.getcwd()); import run_remix_response_dataset",
            ],
            cwd=os.path.abspath(os.path.join(ROOT, "clip_dinoiser")),
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout={result.stdout}\nstderr={result.stderr}",
        )

    def test_build_feature_label_map_uses_schema_field_names(self):
        feature_groups = {
            "quality.laplacian": np.zeros((2, 4), dtype=np.float32),
            "difficulty.small_ratio": np.zeros((2, 5), dtype=np.float32),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            schema_path = os.path.join(tmpdir, "schema.json")
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(TEST_SCHEMA, f, indent=2, ensure_ascii=False)

            labels = build_feature_label_map(feature_groups, schema_path=schema_path)

        self.assertEqual(
            labels["quality.laplacian"],
            ["hist[0]", "hist[1]", "log_num_values", "empty_flag"],
        )
        self.assertEqual(
            labels["difficulty.small_ratio"],
            ["delta_profile[0]", "delta_profile[1]", "log_num_values", "empty_flag", "mass_small_extreme"],
        )

    def test_summarize_portrait_shift_reports_top_blocks_and_features(self):
        delta_phi = {
            "quality.laplacian": [0.1, -0.4, 0.05, 0.0],
            "coverage.knn_local_density": [0.01, 0.02, 0.03],
        }
        label_map = {
            "quality.laplacian": ["hist[0]", "hist[1]", "log_num_values", "empty_flag"],
            "coverage.knn_local_density": ["profile[0]", "profile[1]", "q50"],
        }

        summary = summarize_portrait_shift(delta_phi, label_map, top_blocks=1, top_features_per_block=2)

        self.assertEqual(summary["top_blocks"][0]["block_name"], "quality.laplacian")
        top_features = summary["top_blocks"][0]["top_features"]
        self.assertEqual(top_features[0]["feature"], "hist[1]")
        self.assertAlmostEqual(top_features[0]["delta"], -0.4)


if __name__ == "__main__":
    unittest.main()
