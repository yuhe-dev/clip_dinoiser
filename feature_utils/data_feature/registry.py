from typing import Dict, Callable, Any
from feature_utils.data_feature.implementations.coverage import KNNLocalDensityCLIPFaiss, PrototypeMarginCLIPFaiss
from feature_utils.data_feature.implementations.difficulty import (
    EmpiricalDifficultyMaskClip,
    SemanticAmbiguityCLIP,
    SmallObjectRatioCOCOStuff,
)
from feature_utils.data_feature.implementations.quality import (
    BoundaryGradientAdherence,
    LaplacianSharpness,
    WeakTexturePCANoise
)

_FEATURE_REGISTRY: Dict[str, Callable[..., Any]] = {
    "laplacian": lambda **kw: LaplacianSharpness(),
    "noise_pca": lambda **kw: WeakTexturePCANoise(**kw),
    "small_ratio": lambda **kw: SmallObjectRatioCOCOStuff(**kw),
    "visual_semantic_gap_clip": lambda **kw: SemanticAmbiguityCLIP(**kw),
    "empirical_iou_maskclip": lambda **kw: EmpiricalDifficultyMaskClip(**kw),
    "bga": lambda **kw: BoundaryGradientAdherence(**kw),
    "knn_local_density_faiss": lambda **kw: KNNLocalDensityCLIPFaiss(**kw),
    "prototype_margin_faiss": lambda **kw: PrototypeMarginCLIPFaiss(**kw),
}

def build_feature(feature_name: str, feature_kwargs=None):
    feature_kwargs = feature_kwargs or {}
    key = feature_name.lower().strip()
    if key not in _FEATURE_REGISTRY:
        raise ValueError(f"Unkown feature_name='{feature_name}."
                         f"Available: {list(_FEATURE_REGISTRY.keys())}")
    return _FEATURE_REGISTRY[key](**feature_kwargs)
