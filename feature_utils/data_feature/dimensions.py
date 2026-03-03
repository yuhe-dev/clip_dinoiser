# feature_utils/data_quality/dimensions.py
from .base import BaseMetric
from abc import ABC

class QualityDimension(BaseMetric, ABC):
    def __init__(self, name):
        super().__init__(name)
        self.dim_type = "Quality"

class DifficultyDimension(BaseMetric, ABC):
    def __init__(self, name):
        super().__init__(name)
        self.dim_type = "Difficulty"

class CoverageDimension(BaseMetric, ABC):
    def __init__(self, name):
        super().__init__(name)
        self.dim_type = "Coverage"