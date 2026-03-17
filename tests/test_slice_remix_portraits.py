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
from clip_dinoiser.slice_remix.portraits import (
    compute_portrait_shift,
    compute_slice_portraits,
    load_portrait_feature_groups,
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


if __name__ == "__main__":
    unittest.main()
