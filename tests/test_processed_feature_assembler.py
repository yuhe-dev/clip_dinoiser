import os
import json
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_discovery.assembler import (
    ProcessedFeatureAssembler,
    _ensure_numpy_pickle_compat,
)


TEST_SCHEMA = {
    "schema_version": "processed_feature_bundle.v1",
    "dimensions": {
        "quality": {
            "schema_version": "quality.v1",
            "features": {
                "laplacian": {
                    "model_input_fields": ["hist", "log_num_values", "empty_flag"],
                }
            },
        },
        "difficulty": {
            "schema_version": "difficulty.v1",
            "features": {
                "small_ratio": {
                    "model_input_fields": ["delta_profile", "log_num_values", "empty_flag", "mass_small_extreme"],
                }
            },
        },
        "coverage": {
            "schema_version": "coverage.v1",
            "features": {
                "knn_local_density": {
                    "model_input_fields": ["profile", "q50"],
                }
            },
        },
    },
}


class ProcessedFeatureAssemblerTests(unittest.TestCase):
    def _build_records(self):
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
                        "empty_flag": 1,
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
                        "summary": {
                            "mass_small_extreme": 0.5,
                        },
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
                        "summary": {
                            "mass_small_extreme": 0.7,
                        },
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
                        "summary": {
                            "q50": 0.7,
                        },
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
                        "summary": {
                            "q50": 0.75,
                        },
                    }
                },
            },
        ]
        return quality_records, difficulty_records, coverage_records

    def test_assembler_builds_block_and_flat_views_from_processed_records(self):
        quality_records, difficulty_records, coverage_records = self._build_records()

        assembler = ProcessedFeatureAssembler.from_processed_records(
            quality_records=quality_records,
            difficulty_records=difficulty_records,
            coverage_records=coverage_records,
            schema=TEST_SCHEMA,
        )

        self.assertEqual(assembler.sample_count, 2)
        self.assertEqual(
            assembler.list_blocks(),
            [
                "quality.laplacian",
                "difficulty.small_ratio",
                "coverage.knn_local_density",
            ],
        )
        self.assertEqual(assembler.get_block("quality.laplacian").matrix.shape, (2, 4))
        self.assertEqual(assembler.get_block("difficulty.small_ratio").matrix.shape, (2, 5))
        self.assertEqual(assembler.get_block("coverage.knn_local_density").matrix.shape, (2, 3))
        self.assertEqual(assembler.get_flat_view().shape, (2, 12))

    def test_block_extraction_follows_model_input_field_order(self):
        quality_records, difficulty_records, coverage_records = self._build_records()

        assembler = ProcessedFeatureAssembler.from_processed_records(
            quality_records=quality_records,
            difficulty_records=difficulty_records,
            coverage_records=coverage_records,
            schema=TEST_SCHEMA,
        )

        block = assembler.get_block("difficulty.small_ratio")
        np.testing.assert_allclose(
            block.matrix[0],
            np.asarray([0.2, 0.3, 1.0, 0.0, 0.5], dtype=np.float32),
        )

    def test_alignment_validation_rejects_mismatched_image_ids(self):
        quality_records, difficulty_records, coverage_records = self._build_records()
        difficulty_records[1]["image_rel"] = "images/train2017/9999.jpg"

        with self.assertRaises(ValueError):
            ProcessedFeatureAssembler.from_processed_records(
                quality_records=quality_records,
                difficulty_records=difficulty_records,
                coverage_records=coverage_records,
                schema=TEST_SCHEMA,
            )

    def test_metadata_exposes_block_ranges_and_schema_versions(self):
        quality_records, difficulty_records, coverage_records = self._build_records()

        assembler = ProcessedFeatureAssembler.from_processed_records(
            quality_records=quality_records,
            difficulty_records=difficulty_records,
            coverage_records=coverage_records,
            schema=TEST_SCHEMA,
        )

        metadata = assembler.get_metadata()

        self.assertEqual(metadata["sample_count"], 2)
        self.assertEqual(metadata["schema_version"], "processed_feature_bundle.v1")
        self.assertEqual(metadata["block_order"], assembler.list_blocks())
        self.assertEqual(metadata["block_ranges"]["quality.laplacian"], [0, 4])
        self.assertEqual(metadata["block_ranges"]["difficulty.small_ratio"], [4, 9])
        self.assertEqual(metadata["block_ranges"]["coverage.knn_local_density"], [9, 12])
        self.assertEqual(metadata["dimension_schema_versions"]["quality"], "quality.v1")

    def test_save_and_load_round_trip_preserves_flat_and_block_views(self):
        quality_records, difficulty_records, coverage_records = self._build_records()

        assembler = ProcessedFeatureAssembler.from_processed_records(
            quality_records=quality_records,
            difficulty_records=difficulty_records,
            coverage_records=coverage_records,
            schema=TEST_SCHEMA,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            assembler.save(tmpdir)
            restored = ProcessedFeatureAssembler.load(tmpdir)

            np.testing.assert_allclose(restored.get_flat_view(), assembler.get_flat_view())
            self.assertEqual(restored.sample_ids, assembler.sample_ids)
            self.assertEqual(restored.list_blocks(), assembler.list_blocks())
            self.assertEqual(restored.get_metadata()["block_ranges"], assembler.get_metadata()["block_ranges"])

    def test_from_processed_paths_loads_saved_dimension_bundles(self):
        quality_records, difficulty_records, coverage_records = self._build_records()

        with tempfile.TemporaryDirectory() as tmpdir:
            quality_path = os.path.join(tmpdir, "quality_processed_features.npy")
            difficulty_path = os.path.join(tmpdir, "difficulty_processed_features.npy")
            coverage_path = os.path.join(tmpdir, "coverage_processed_features.npy")
            schema_path = os.path.join(tmpdir, "schema.json")

            np.save(quality_path, np.asarray(quality_records, dtype=object), allow_pickle=True)
            np.save(difficulty_path, np.asarray(difficulty_records, dtype=object), allow_pickle=True)
            np.save(coverage_path, np.asarray(coverage_records, dtype=object), allow_pickle=True)
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(TEST_SCHEMA, f, indent=2, ensure_ascii=False)

            assembler = ProcessedFeatureAssembler.from_processed_paths(
                quality_path=quality_path,
                difficulty_path=difficulty_path,
                coverage_path=coverage_path,
                schema_path=schema_path,
            )

            self.assertEqual(assembler.sample_count, 2)
            self.assertEqual(assembler.list_blocks()[0], "quality.laplacian")

    def test_numpy_pickle_compat_registers_numpy_core_aliases(self):
        _ensure_numpy_pickle_compat()

        self.assertIn("numpy._core", sys.modules)
        self.assertIn("numpy._core.multiarray", sys.modules)

    def test_debug_summary_reports_block_and_flat_statistics(self):
        quality_records, difficulty_records, coverage_records = self._build_records()

        assembler = ProcessedFeatureAssembler.from_processed_records(
            quality_records=quality_records,
            difficulty_records=difficulty_records,
            coverage_records=coverage_records,
            schema=TEST_SCHEMA,
        )

        summary = assembler.get_debug_summary()

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["flat_shape"], [2, 12])
        self.assertTrue(summary["flat_all_finite"])
        self.assertEqual(summary["blocks"]["quality.laplacian"]["shape"], [2, 4])
        self.assertTrue(summary["blocks"]["quality.laplacian"]["all_finite"])


if __name__ == "__main__":
    unittest.main()
