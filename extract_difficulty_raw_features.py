import argparse
import importlib.util
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from feature_utils.data_feature.bundle import RawBundleIO, RawFeatureBundle, build_raw_feature_stats
from feature_utils.data_feature.extraction import DifficultyRawExtractor


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
    return build_raw_feature_stats(records=records, feature_keys=DIFFICULTY_FEATURE_KEYS)


def save_difficulty_feature_bundle(
    output_root: str,
    records: Sequence[Dict[str, object]],
    stats: Dict[str, object],
    subset_root: str,
    index_path: str,
    feature_meta: Dict[str, object],
    class_names: Sequence[str],
) -> Tuple[str, str, str]:
    bundle = RawFeatureBundle(
        dimension_name="difficulty",
        records=list(records),
        stats=stats,
        feature_config={
            "subset_root": subset_root,
            "index_path": index_path,
            "feature_meta": feature_meta,
            "class_names_count": int(len(class_names)),
            "records_file": "difficulty_raw_features.npy",
            "stats_file": "difficulty_global_stats.json",
        },
    )
    paths = RawBundleIO().save(bundle, output_root)
    return paths["records_path"], paths["stats_path"], paths["config_path"]


def extract_difficulty_records(
    subset_root: str,
    subset_records: Sequence[Dict[str, object]],
    class_names: Sequence[str],
    feature_meta: Dict[str, object],
    show_progress: bool = True,
) -> List[Dict[str, object]]:
    extractor = DifficultyRawExtractor()
    extractor_feature_meta = dict(feature_meta)
    extractor_feature_meta["class_names"] = list(class_names)
    return extractor.extract_records(
        subset_root=subset_root,
        subset_records=subset_records,
        feature_meta=extractor_feature_meta,
        show_progress=show_progress,
    )


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
