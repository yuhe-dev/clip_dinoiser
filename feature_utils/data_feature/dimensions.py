# feature_utils/data_quality/dimensions.py
from .base import BaseMetric
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np

class QualityDimension(BaseMetric, ABC):
    def __init__(self, name):
        super().__init__(name)
        self.dim_type = "Quality"

    @abstractmethod
    def get_vector_score(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        pass

class DifficultyDimension(BaseMetric, ABC):
    def __init__(self, name):
        super().__init__(name)
        self.dim_type = "Difficulty"

    @abstractmethod
    def get_vector_score(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        pass

class CoverageDimension(BaseMetric, ABC):
    def __init__(self, name):
        super().__init__(name)
        self.dim_type = "Coverage"

    @abstractmethod
    def get_vector_score(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        pass
