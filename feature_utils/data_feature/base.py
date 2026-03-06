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

    @abstractmethod
    def get_vector_score(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        向量化特征接口：将样本映射到固定长度或约定长度的特征向量。
        具体维度与构造策略由各子类定义。
        """
        pass
