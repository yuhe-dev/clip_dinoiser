import os

import numpy as np

from .base import BaseRawExtractor


class QualityRawExtractor(BaseRawExtractor):
    dimension_name = "quality"

    def __init__(self, feature_factory=None):
        if feature_factory is None:
            from feature_utils.data_feature.implementations.quality import (
                BoundaryGradientAdherence,
                LaplacianSharpness,
                WeakTexturePCANoise,
            )

            def feature_factory(feature_meta):
                return {
                    "laplacian": LaplacianSharpness(),
                    "noise_pca": WeakTexturePCANoise(
                        patch_size=int(feature_meta.get("patch_size", 8)),
                        stride=int(feature_meta.get("stride", 8)),
                    ),
                    "bga": BoundaryGradientAdherence(),
                }

        super().__init__(feature_factory=feature_factory)

    def load_sample_context(self, subset_root: str, record: dict) -> dict:
        import cv2

        image_path = os.path.join(subset_root, str(record["image_rel"]))
        annotation_path = os.path.join(subset_root, str(record["annotation_rel"]))
        image = cv2.imread(image_path)
        mask = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            return None
        return {"image": image, "mask": mask}

    def extract_single_record(
        self,
        record: dict,
        sample_context: dict,
        feature_instances: dict,
        feature_meta: dict,
    ) -> dict:
        image = sample_context["image"]
        mask = sample_context["mask"]
        return {
            "image_rel": str(record["image_rel"]),
            "annotation_rel": str(record["annotation_rel"]),
            "laplacian_raw": np.asarray(
                feature_instances["laplacian"].get_vector_score(image, meta=feature_meta),
                dtype=np.float32,
            ),
            "noise_pca_raw": np.asarray(
                feature_instances["noise_pca"].get_vector_score(image, meta=feature_meta),
                dtype=np.float32,
            ),
            "bga_raw": np.asarray(
                feature_instances["bga"].get_vector_score(image, mask=mask, meta=feature_meta),
                dtype=np.float32,
            ),
        }
