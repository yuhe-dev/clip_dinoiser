import argparse
import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


QUALITY_FEATURE_KEYS: Tuple[str, ...] = ("laplacian_raw", "noise_pca_raw", "bga_raw")


def load_subset_records(index_path: str) -> List[Dict[str, object]]:
    records = np.load(index_path, allow_pickle=True)
    return [dict(item) for item in records.tolist()]


def compute_global_stats(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    stats: Dict[str, object] = {
        "num_samples": int(len(records)),
        "features": {},
    }

    for feature_key in QUALITY_FEATURE_KEYS:
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


def save_quality_feature_bundle(
    output_root: str,
    records: Sequence[Dict[str, object]],
    stats: Dict[str, object],
    subset_root: str,
    index_path: str,
    feature_meta: Dict[str, object],
) -> Tuple[str, str, str]:
    os.makedirs(output_root, exist_ok=True)

    records_path = os.path.join(output_root, "quality_raw_features.npy")
    stats_path = os.path.join(output_root, "quality_global_stats.json")
    config_path = os.path.join(output_root, "quality_feature_config.json")

    np.save(records_path, np.asarray(list(records), dtype=object), allow_pickle=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    config = {
        "subset_root": subset_root,
        "index_path": index_path,
        "feature_meta": feature_meta,
        "records_file": os.path.basename(records_path),
        "stats_file": os.path.basename(stats_path),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return records_path, stats_path, config_path


def extract_quality_records(
    subset_root: str,
    subset_records: Sequence[Dict[str, object]],
    feature_meta: Dict[str, object],
    show_progress: bool = True,
) -> List[Dict[str, object]]:
    import cv2
    from tqdm import tqdm

    from clip_dinoiser.feature_utils.data_feature.implementations.quality import (
        BoundaryGradientAdherence,
        LaplacianSharpness,
        WeakTexturePCANoise,
    )

    laplacian = LaplacianSharpness()
    noise = WeakTexturePCANoise(
        patch_size=int(feature_meta.get("patch_size", 8)),
        stride=int(feature_meta.get("stride", 8)),
    )
    bga = BoundaryGradientAdherence()

    iterator = tqdm(
        subset_records,
        desc="Extracting quality raw features",
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

        extracted.append(
            {
                "image_rel": image_rel,
                "annotation_rel": annotation_rel,
                "laplacian_raw": laplacian.get_vector_score(image, meta=feature_meta).astype(np.float32),
                "noise_pca_raw": noise.get_vector_score(image, meta=feature_meta).astype(np.float32),
                "bga_raw": bga.get_vector_score(image, mask=mask, meta=feature_meta).astype(np.float32),
            }
        )
    return extracted


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract raw quality features for a COCO-Stuff subset.")
    parser.add_argument("--subset-root", default="./data/coco_stuff50k")
    parser.add_argument("--index-path", default=None, help="Defaults to <subset-root>/sample_index.npy")
    parser.add_argument("--output-root", default="./data/data_feature/quality")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    subset_root = os.path.abspath(args.subset_root)
    index_path = os.path.abspath(args.index_path or os.path.join(subset_root, "sample_index.npy"))
    output_root = os.path.abspath(args.output_root)
    feature_meta = {"patch_size": args.patch_size, "stride": args.stride}

    print(f"[quality-extract] subset_root={subset_root}")
    print(f"[quality-extract] index_path={index_path}")
    print(f"[quality-extract] output_root={output_root}")
    print(f"[quality-extract] feature_meta={feature_meta}")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Subset index not found: {index_path}")

    subset_records = load_subset_records(index_path)
    print(f"[quality-extract] loaded {len(subset_records)} subset records")

    records = extract_quality_records(
        subset_root=subset_root,
        subset_records=subset_records,
        feature_meta=feature_meta,
        show_progress=not args.no_progress,
    )
    print(f"[quality-extract] extracted {len(records)} quality feature records")

    stats = compute_global_stats(records)
    records_path, stats_path, config_path = save_quality_feature_bundle(
        output_root=output_root,
        records=records,
        stats=stats,
        subset_root=subset_root,
        index_path=index_path,
        feature_meta=feature_meta,
    )

    print(f"[quality-extract] saved records: {records_path}")
    print(f"[quality-extract] saved stats: {stats_path}")
    print(f"[quality-extract] saved config: {config_path}")
    preview = records[:2]
    print(f"[quality-extract] preview: {preview}")


if __name__ == "__main__":
    main()
