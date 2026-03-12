import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.feature_utils.data_feature.extraction.difficulty import DifficultyRawExtractor


class _SmallRatioFeature:
    def get_profile_and_count(self, image, mask=None, meta=None):
        return np.asarray([0.0, 0.4, 1.0], dtype=np.float32), 3


class _ArrayFeature:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def get_vector_score(self, image, mask=None, meta=None):
        return self.values.copy()


class _TestingDifficultyExtractor(DifficultyRawExtractor):
    def __init__(self):
        super().__init__(
            feature_factory=lambda feature_meta: {
                "small_ratio": _SmallRatioFeature(),
                "visual_semantic_gap": _ArrayFeature([0.2, 0.8]),
                "empirical_iou": _ArrayFeature([0.1, 0.7]),
            }
        )

    def load_sample_context(self, subset_root, record):
        return {
            "image": np.zeros((4, 4, 3), dtype=np.uint8),
            "mask": np.zeros((4, 4), dtype=np.uint8),
            "meta": {"class_names": ["cat", "dog"], "ignore_index": 255},
        }


class TestDifficultyRawExtractor(unittest.TestCase):
    def test_difficulty_raw_extractor_preserves_small_ratio_num_values(self):
        extractor = _TestingDifficultyExtractor()
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
        self.assertIn("small_ratio_raw", records[0])
        self.assertIn("small_ratio_num_values", records[0])
        self.assertEqual(records[0]["small_ratio_num_values"], 3)
        np.testing.assert_allclose(records[0]["visual_semantic_gap_raw"], np.asarray([0.2, 0.8], dtype=np.float32))
        np.testing.assert_allclose(records[0]["empirical_iou_raw"], np.asarray([0.1, 0.7], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
