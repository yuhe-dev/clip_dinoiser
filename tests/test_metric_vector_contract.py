import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from clip_dinoiser.feature_utils.data_feature.base import BaseMetric
from clip_dinoiser.feature_utils.data_feature.dimensions import (
    CoverageDimension,
    DifficultyDimension,
    QualityDimension,
)


class TestMetricVectorContract(unittest.TestCase):
    def test_base_metric_declares_vector_abstract_method(self):
        self.assertIn("get_vector_score", BaseMetric.__abstractmethods__)

    def test_dimension_classes_keep_vector_method_abstract(self):
        self.assertIn("get_vector_score", QualityDimension.__abstractmethods__)
        self.assertIn("get_vector_score", DifficultyDimension.__abstractmethods__)
        self.assertIn("get_vector_score", CoverageDimension.__abstractmethods__)


if __name__ == "__main__":
    unittest.main()
