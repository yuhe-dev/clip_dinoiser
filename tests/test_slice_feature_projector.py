import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_discovery.assembler import ProcessedFeatureAssembler
from clip_dinoiser.slice_discovery.projector import SliceFeatureProjector
from tests.test_processed_feature_assembler import TEST_SCHEMA


class SliceFeatureProjectorTests(unittest.TestCase):
    def _build_assembler(self):
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
                        "log_num_values": 3.0,
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
                        "log_num_values": 2.0,
                        "empty_flag": 0,
                        "summary": {"mass_small_extreme": 0.7},
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
        ]
        return ProcessedFeatureAssembler.from_processed_records(
            quality_records=quality_records,
            difficulty_records=difficulty_records,
            coverage_records=coverage_records,
            schema=TEST_SCHEMA,
        )

    def test_projector_scales_scalar_fields_but_preserves_distribution_vectors(self):
        assembler = self._build_assembler()
        projector = SliceFeatureProjector(
            scalar_scaler="zscore",
            block_weighting="none",
            pca_components=None,
        )

        result = projector.fit_transform(assembler)

        self.assertEqual(result.matrix.shape, (2, 12))
        self.assertEqual(result.block_ranges["quality.laplacian"], (0, 4))
        np.testing.assert_allclose(result.matrix[:, 0], np.asarray([0.1, 0.3], dtype=np.float32))
        np.testing.assert_allclose(result.matrix[:, 1], np.asarray([0.2, 0.4], dtype=np.float32))
        np.testing.assert_allclose(result.matrix[:, 2], np.asarray([-1.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(result.matrix[:, 3], np.asarray([0.0, 1.0], dtype=np.float32))

    def test_equal_by_block_weighting_scales_each_block_by_inverse_root_width(self):
        assembler = self._build_assembler()
        projector = SliceFeatureProjector(
            scalar_scaler="none",
            block_weighting="equal_by_block",
            pca_components=None,
        )

        result = projector.fit_transform(assembler)

        self.assertAlmostEqual(float(result.matrix[0, 0]), 0.1 / np.sqrt(4), places=6)
        self.assertAlmostEqual(float(result.matrix[0, 4]), 0.2 / np.sqrt(5), places=6)
        self.assertAlmostEqual(float(result.matrix[0, 9]), 0.6 / np.sqrt(3), places=6)

    def test_projected_features_can_be_saved_loaded_and_summarized(self):
        assembler = self._build_assembler()
        projector = SliceFeatureProjector(
            scalar_scaler="zscore",
            block_weighting="equal_by_block",
            pca_components=None,
        )

        projected = projector.fit_transform(assembler)

        with tempfile.TemporaryDirectory() as tmpdir:
            projector.save(projected, tmpdir)
            restored = projector.load(tmpdir)

            np.testing.assert_allclose(restored.matrix, projected.matrix)
            self.assertEqual(restored.sample_ids, projected.sample_ids)
            self.assertEqual(restored.block_ranges, projected.block_ranges)

            summary = projector.get_debug_summary(restored)
            self.assertEqual(summary["matrix_shape"], [2, 12])
            self.assertTrue(summary["all_finite"])
            self.assertEqual(summary["blocks"]["quality.laplacian"]["shape"], [2, 4])


if __name__ == "__main__":
    unittest.main()
