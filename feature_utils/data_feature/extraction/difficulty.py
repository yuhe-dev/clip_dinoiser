import os

import numpy as np

from .base import BaseRawExtractor


class DifficultyRawExtractor(BaseRawExtractor):
    dimension_name = "difficulty"

    def __init__(self, feature_factory=None):
        if feature_factory is None:
            from feature_utils.data_feature.implementations.difficulty import (
                EmpiricalDifficultyMaskClip,
                SemanticAmbiguityCLIP,
                SmallObjectRatioCOCOStuff,
            )
            from open_clip import create_model_from_pretrained, get_tokenizer

            def feature_factory(feature_meta):
                thresholds = feature_meta.get("small_ratio_thresholds", None)
                small_ratio = SmallObjectRatioCOCOStuff(
                    thresholds=list(thresholds) if thresholds is not None else None,
                )
                clip_model_name = str(feature_meta.get("clip_model", "ViT-B-16"))
                clip_pretrained = str(feature_meta.get("clip_pretrained", "laion2b_s34b_b88k"))
                clip_model, clip_preprocess = create_model_from_pretrained(clip_model_name, pretrained=clip_pretrained)
                clip_model.eval()
                clip_device = str(feature_meta.get("clip_device", feature_meta.get("device", "cuda")))
                clip_model = clip_model.to(clip_device)
                tokenizer = get_tokenizer(clip_model_name)
                semantic_gap = SemanticAmbiguityCLIP(
                    clip_model=clip_model,
                    tokenizer=tokenizer,
                    preprocess=clip_preprocess,
                    device=clip_device,
                    default_ignore_index=int(feature_meta.get("ignore_index", 255)),
                    use_things_only=bool(feature_meta.get("use_things_only", False)),
                    min_region_pixels=int(feature_meta.get("min_region_pixels", 256)),
                    max_regions_per_image=int(feature_meta.get("max_regions_per_image", 20)),
                )
                empirical_iou = EmpiricalDifficultyMaskClip(
                    model_cfg=str(feature_meta.get("model_cfg", "configs/maskclip.yaml")),
                    class_names=list(feature_meta.get("class_names", [])),
                    device=str(feature_meta.get("maskclip_device", feature_meta.get("device", "cuda"))),
                    default_ignore_index=int(feature_meta.get("ignore_index", 255)),
                )
                return {
                    "small_ratio": small_ratio,
                    "visual_semantic_gap": semantic_gap,
                    "empirical_iou": empirical_iou,
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
        return {
            "image": image,
            "mask": mask,
            "meta": {
                "class_names": list(record.get("class_names", [])),
                "ignore_index": int(record.get("ignore_index", 255)),
                "use_things_only": bool(record.get("use_things_only", False)),
            },
        }

    def extract_single_record(
        self,
        record: dict,
        sample_context: dict,
        feature_instances: dict,
        feature_meta: dict,
    ) -> dict:
        image = sample_context["image"]
        mask = sample_context["mask"]
        meta = dict(sample_context.get("meta", {}))
        if "class_names" not in meta or not meta["class_names"]:
            meta["class_names"] = list(feature_meta.get("class_names", []))
        meta["ignore_index"] = int(meta.get("ignore_index", feature_meta.get("ignore_index", 255)))
        meta["use_things_only"] = bool(meta.get("use_things_only", feature_meta.get("use_things_only", False)))

        small_ratio_profile, small_ratio_count = feature_instances["small_ratio"].get_profile_and_count(
            image,
            mask=mask,
            meta=meta,
        )
        return {
            "image_rel": str(record["image_rel"]),
            "annotation_rel": str(record["annotation_rel"]),
            "small_ratio_raw": np.asarray(small_ratio_profile, dtype=np.float32),
            "small_ratio_num_values": int(small_ratio_count),
            "visual_semantic_gap_raw": np.asarray(
                feature_instances["visual_semantic_gap"].get_vector_score(image, mask=mask, meta=meta),
                dtype=np.float32,
            ),
            "empirical_iou_raw": np.asarray(
                feature_instances["empirical_iou"].get_vector_score(image, mask=mask, meta=meta),
                dtype=np.float32,
            ),
        }
