import os

import numpy as np

from .base import BaseRawExtractor


class CoverageRawExtractor(BaseRawExtractor):
    dimension_name = "coverage"

    def __init__(self, feature_factory=None):
        if feature_factory is None:
            from feature_utils.data_feature.implementations.coverage import (
                KNNLocalDensityCLIPFaiss,
                PrototypeMarginCLIPFaiss,
            )

            def feature_factory(feature_meta):
                embedding_root = str(feature_meta["embedding_root"])
                return {
                    "knn_local_density": KNNLocalDensityCLIPFaiss(
                        cache_dir=embedding_root,
                        emb_file=str(feature_meta.get("embeddings_file", "visual_emb.npy")),
                        paths_file=str(feature_meta.get("paths_file", "clip_paths_abs.json")),
                        k=int(feature_meta.get("knn_k", 50)),
                        metric=str(feature_meta.get("knn_metric", "cosine")),
                        mode="mean_dist",
                        include_self=bool(feature_meta.get("include_self", False)),
                        normalize_for_cosine=bool(feature_meta.get("normalize_for_cosine", True)),
                    ),
                    "prototype_distance": PrototypeMarginCLIPFaiss(
                        cache_dir=embedding_root,
                        emb_file=str(feature_meta.get("embeddings_file", "visual_emb.npy")),
                        paths_file=str(feature_meta.get("paths_file", "clip_paths_abs.json")),
                        centroid_file=str(feature_meta.get("centroid_file", "prototypes_k200.npy")),
                        top_m=int(feature_meta.get("prototype_top_m", 8)),
                        normalize=bool(feature_meta.get("normalize_for_cosine", True)),
                    ),
                }

        super().__init__(feature_factory=feature_factory)

    def load_sample_context(self, subset_root: str, record: dict) -> dict:
        image_path = os.path.abspath(os.path.join(subset_root, str(record["image_rel"])))
        return {"meta": {"img_path": image_path, "path": image_path}}

    def extract_single_record(
        self,
        record: dict,
        sample_context: dict,
        feature_instances: dict,
        feature_meta: dict,
    ) -> dict:
        del feature_meta
        meta = dict(sample_context["meta"])
        return {
            "image_rel": str(record["image_rel"]),
            "annotation_rel": str(record.get("annotation_rel", "")),
            "knn_neighbor_distances_raw": np.asarray(
                feature_instances["knn_local_density"].get_vector_score(None, meta=meta),
                dtype=np.float32,
            ),
            "prototype_distances_raw": np.asarray(
                feature_instances["prototype_distance"].get_vector_score(None, meta=meta),
                dtype=np.float32,
            ),
        }
