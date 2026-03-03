# feature_utils/data_quality/base.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Optional

class BaseMetric(ABC):
    """所有数据描述指标的基类"""
    def __init__(self, name):
        self.name = name
        self.dim_type: Optional[str] = None

    @abstractmethod
    def get_score(self, image: np.ndarray, mask: Optional[np.ndarray] = None, meta: Optional[Dict[str, Any]] = None) -> float:
        """
        核心方法：计算得分
        image: BGR 图像
        mask: 对应的真值掩码 (部分指标如 Laplacian 可能不需要)
        """
        pass