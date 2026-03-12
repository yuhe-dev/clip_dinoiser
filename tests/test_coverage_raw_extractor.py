import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.feature_utils.data_feature.extraction.coverage import CoverageRawExtractor


class _ArrayFeature:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def get_vector_score(self, image, mask=None, meta=None):
        return self.values.copy()


class _TestingCoverageExtractor(CoverageRawExtractor):
    def __init__(self):
        super().__init__(
            feature_factory=lambda feature_meta: {
                "knn_local_density": _ArrayFeature([0.1, 0.2]),
                "prototype_distance": _ArrayFeature([0.3, 0.4, 0.5]),
            }
        )

    def load_sample_context(self, subset_root, record):
        image_path = os.path.abspath(os.path.join(subset_root, str(record["image_rel"])))
        return {"meta": {"img_path": image_path, "path": image_path}}


class TestCoverageRawExtractor(unittest.TestCase):
    def test_coverage_raw_extractor_emits_existing_raw_field_names(self):
        extractor = _TestingCoverageExtractor()
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
        self.assertIn("knn_neighbor_distances_raw", records[0])
        self.assertIn("prototype_distances_raw", records[0])
        np.testing.assert_allclose(records[0]["knn_neighbor_distances_raw"], np.asarray([0.1, 0.2], dtype=np.float32))
        np.testing.assert_allclose(records[0]["prototype_distances_raw"], np.asarray([0.3, 0.4, 0.5], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
