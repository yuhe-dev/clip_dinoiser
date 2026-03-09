import argparse
import importlib.util
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np


DIFFICULTY_FEATURE_KEYS: Tuple[str, ...] = (
    "small_ratio_raw",
    "visual_semantic_gap_raw",
    "empirical_iou_raw",
)


def load_subset_records(index_path: str) -> List[Dict[str, object]]:
    records = np.load(index_path, allow_pickle=True)
    return [dict(item) for item in records.tolist()]


def limit_subset_records(records: Sequence[Dict[str, object]], limit: int = None) -> List[Dict[str, object]]:
    if limit is None or int(limit) <= 0:
        return list(records)
    return list(records[: int(limit)])


def load_coco_stuff_classes(dataset_module_path: str) -> List[str]:
    spec = importlib.util.spec_from_file_location("coco_stuff_dataset_module", dataset_module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load dataset module from: {dataset_module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset_cls = getattr(module, "COCOStuffDataset", None)
    if dataset_cls is None or not hasattr(dataset_cls, "CLASSES"):
        raise AttributeError("COCOStuffDataset.CLASSES not found in dataset module.")
    return list(dataset_cls.CLASSES)


def compute_global_stats(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    stats: Dict[str, object] = {"num_samples": int(len(records)), "features": {}}
    for feature_key in DIFFICULTY_FEATURE_KEYS:
        arrays = [
            np.asarray(record.get(feature_key, np.asarray([], dtype=np.float32)), dtype=np.float32)
            for record in records
        ]
        lengths = [int(arr.size) for arr in arrays]
        non_empty = [arr for arr in arrays if arr.size > 0]
        if non_empty:
            merged = np.concatenate(non_empty, axis=0)
            feature_stats = {
                "global_min": float(merged.min()),
                "global_max": float(merged.max()),
                "global_mean": float(merged.mean()),
                "global_std": float(merged.std()),
                "total_values": int(merged.size),
            }
        else:
            feature_stats = {
                "global_min": None,
                "global_max": None,
                "global_mean": None,
                "global_std": None,
                "total_values": 0,
            }
        feature_stats.update(
            {
                "empty_samples": int(sum(1 for length in lengths if length == 0)),
                "length_min": int(min(lengths)) if lengths else 0,
                "length_max": int(max(lengths)) if lengths else 0,
                "length_mean": float(np.mean(lengths)) if lengths else 0.0,
            }
        )
        stats["features"][feature_key] = feature_stats
    return stats


def save_difficulty_feature_bundle(
    output_root: str,
    records: Sequence[Dict[str, object]],
    stats: Dict[str, object],
    subset_root: str,
    index_path: str,
    feature_meta: Dict[str, object],
    class_names: Sequence[str],
) -> Tuple[str, str, str]:
    os.makedirs(output_root, exist_ok=True)

    records_path = os.path.join(output_root, "difficulty_raw_features.npy")
    stats_path = os.path.join(output_root, "difficulty_global_stats.json")
    config_path = os.path.join(output_root, "difficulty_feature_config.json")

    np.save(records_path, np.asarray(list(records), dtype=object), allow_pickle=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    config = {
        "subset_root": subset_root,
        "index_path": index_path,
        "feature_meta": feature_meta,
        "class_names_count": int(len(class_names)),
        "records_file": os.path.basename(records_path),
        "stats_file": os.path.basename(stats_path),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return records_path, stats_path, config_path


def extract_difficulty_records(
    subset_root: str,
    subset_records: Sequence[Dict[str, object]],
    class_names: Sequence[str],
    feature_meta: Dict[str, object],
    show_progress: bool = True,
) -> List[Dict[str, object]]:
    import cv2
    from open_clip import create_model_from_pretrained, get_tokenizer
    from tqdm import tqdm

    from clip_dinoiser.feature_utils.data_feature.implementations.difficulty import (
        EmpiricalDifficultyMaskClip,
        SemanticAmbiguityCLIP,
        SmallObjectRatioCOCOStuff,
    )

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
        class_names=list(class_names),
        device=str(feature_meta.get("maskclip_device", feature_meta.get("device", "cuda"))),
        default_ignore_index=int(feature_meta.get("ignore_index", 255)),
    )

    iterator = tqdm(
        subset_records,
        desc="Extracting difficulty raw features",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    extracted: List[Dict[str, object]] = []
    for record in iterator:
        image_rel = str(record["image_rel"])
        annotation_rel = str(record["annotation_rel"])
        image_path = os.path.join(subset_root, image_rel)
        annotation_path = os.path.join(subset_root, annotation_rel)

        image = cv2.imread(image_path)
        mask = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue

        meta = {
            "class_names": list(class_names),
            "ignore_index": int(feature_meta.get("ignore_index", 255)),
            "use_things_only": bool(feature_meta.get("use_things_only", False)),
        }
        extracted.append(
            {
                "image_rel": image_rel,
                "annotation_rel": annotation_rel,
                "small_ratio_raw": small_ratio.get_vector_score(image, mask=mask, meta=meta).astype(np.float32),
                "visual_semantic_gap_raw": semantic_gap.get_vector_score(image, mask=mask, meta=meta).astype(np.float32),
                "empirical_iou_raw": empirical_iou.get_vector_score(image, mask=mask, meta=meta).astype(np.float32),
            }
        )
    return extracted


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract raw difficulty features for a COCO-Stuff subset.")
    parser.add_argument("--subset-root", default="./data/coco_stuff50k")
    parser.add_argument("--index-path", default=None, help="Defaults to <subset-root>/sample_index.npy")
    parser.add_argument("--output-root", default="./data/data_feature/difficulty")
    parser.add_argument(
        "--dataset-module-path",
        default="./segmentation/datasets/coco_stuff.py",
        help="Path to the COCO-Stuff dataset definition containing COCOStuffDataset.CLASSES.",
    )
    parser.add_argument("--model-cfg", default="configs/maskclip.yaml")
    parser.add_argument("--clip-model", default="ViT-B-16")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip-device", default=None)
    parser.add_argument("--maskclip-device", default=None)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--min-region-pixels", type=int, default=256)
    parser.add_argument("--max-regions-per-image", type=int, default=20)
    parser.add_argument("--use-things-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="If > 0, only process the first N subset records.")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    subset_root = os.path.abspath(args.subset_root)
    index_path = os.path.abspath(args.index_path or os.path.join(subset_root, "sample_index.npy"))
    output_root = os.path.abspath(args.output_root)
    dataset_module_path = os.path.abspath(args.dataset_module_path)
    class_names = load_coco_stuff_classes(dataset_module_path)
    feature_meta = {
        "model_cfg": args.model_cfg,
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "device": args.device,
        "clip_device": args.clip_device or args.device,
        "maskclip_device": args.maskclip_device or args.device,
        "ignore_index": args.ignore_index,
        "min_region_pixels": args.min_region_pixels,
        "max_regions_per_image": args.max_regions_per_image,
        "use_things_only": bool(args.use_things_only),
    }

    print(f"[difficulty-extract] subset_root={subset_root}")
    print(f"[difficulty-extract] index_path={index_path}")
    print(f"[difficulty-extract] output_root={output_root}")
    print(f"[difficulty-extract] dataset_module_path={dataset_module_path}")
    print(f"[difficulty-extract] class_names_count={len(class_names)}")
    print(f"[difficulty-extract] feature_meta={feature_meta}")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Subset index not found: {index_path}")

    subset_records = load_subset_records(index_path)
    total_records = len(subset_records)
    subset_records = limit_subset_records(subset_records, limit=args.limit)
    print(f"[difficulty-extract] loaded {total_records} subset records")
    print(f"[difficulty-extract] processing {len(subset_records)} subset records (limit={args.limit})")

    records = extract_difficulty_records(
        subset_root=subset_root,
        subset_records=subset_records,
        class_names=class_names,
        feature_meta=feature_meta,
        show_progress=not args.no_progress,
    )
    print(f"[difficulty-extract] extracted {len(records)} difficulty feature records")

    stats = compute_global_stats(records)
    records_path, stats_path, config_path = save_difficulty_feature_bundle(
        output_root=output_root,
        records=records,
        stats=stats,
        subset_root=subset_root,
        index_path=index_path,
        feature_meta=feature_meta,
        class_names=class_names,
    )

    print(f"[difficulty-extract] saved records: {records_path}")
    print(f"[difficulty-extract] saved stats: {stats_path}")
    print(f"[difficulty-extract] saved config: {config_path}")
    print(f"[difficulty-extract] preview: {records[:2]}")


if __name__ == "__main__":
    main()
