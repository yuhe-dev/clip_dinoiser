import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.feature_utils.data_feature.extraction.quality import QualityRawExtractor


class _ArrayFeature:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def get_vector_score(self, image, mask=None, meta=None):
        return self.values.copy()


class _TestingQualityExtractor(QualityRawExtractor):
    def __init__(self):
        super().__init__(
            feature_factory=lambda feature_meta: {
                "laplacian": _ArrayFeature([1.0, 2.0]),
                "noise_pca": _ArrayFeature([3.0]),
                "bga": _ArrayFeature([0.5, 0.8]),
            }
        )

    def load_sample_context(self, subset_root, record):
        return {
            "image": np.zeros((4, 4, 3), dtype=np.uint8),
            "mask": np.zeros((4, 4), dtype=np.uint8),
        }


class TestQualityRawExtractor(unittest.TestCase):
    def test_quality_raw_extractor_emits_existing_raw_field_names(self):
        extractor = _TestingQualityExtractor()
        records = extractor.extract_records(
            subset_root="unused",
            subset_records=[
                {
                    "image_rel": "images/train2017/0001.jpg",
                    "annotation_rel": "annotations/train2017/0001_labelTrainIds.png",
                }
            ],
            feature_meta={},
            show_progress=False,
        )

        self.assertEqual(len(records), 1)
        self.assertIn("laplacian_raw", records[0])
        self.assertIn("noise_pca_raw", records[0])
        self.assertIn("bga_raw", records[0])
        np.testing.assert_allclose(records[0]["laplacian_raw"], np.asarray([1.0, 2.0], dtype=np.float32))
        np.testing.assert_allclose(records[0]["noise_pca_raw"], np.asarray([3.0], dtype=np.float32))
        np.testing.assert_allclose(records[0]["bga_raw"], np.asarray([0.5, 0.8], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
