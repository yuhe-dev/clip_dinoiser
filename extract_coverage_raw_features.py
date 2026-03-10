import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np


COVERAGE_FEATURE_KEYS: Tuple[str, ...] = (
    "knn_neighbor_distances_raw",
    "prototype_distances_raw",
)


def load_subset_records(index_path: str) -> List[Dict[str, object]]:
    records = np.load(index_path, allow_pickle=True)
    return [dict(item) for item in records.tolist()]


def limit_subset_records(records: Sequence[Dict[str, object]], limit: int = None) -> List[Dict[str, object]]:
    if limit is None or int(limit) <= 0:
        return list(records)
    return list(records[: int(limit)])


def compute_coverage_global_stats(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    stats: Dict[str, object] = {"num_samples": int(len(records)), "features": {}}
    for feature_key in COVERAGE_FEATURE_KEYS:
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


def save_coverage_feature_bundle(
    output_root: str,
    records: Sequence[Dict[str, object]],
    stats: Dict[str, object],
    subset_root: str,
    index_path: str,
    feature_meta: Dict[str, object],
) -> Tuple[str, str, str]:
    os.makedirs(output_root, exist_ok=True)

    records_path = os.path.join(output_root, "coverage_raw_features.npy")
    stats_path = os.path.join(output_root, "coverage_global_stats.json")
    config_path = os.path.join(output_root, "coverage_feature_config.json")

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


def extract_coverage_records(
    subset_root: str,
    subset_records: Sequence[Dict[str, object]],
    embedding_root: str,
    feature_meta: Dict[str, object],
    show_progress: bool = True,
) -> List[Dict[str, object]]:
    from tqdm import tqdm

    from feature_utils.data_feature.implementations.coverage import (
        KNNLocalDensityCLIPFaiss,
        PrototypeMarginCLIPFaiss,
    )

    knn_feature = KNNLocalDensityCLIPFaiss(
        cache_dir=embedding_root,
        emb_file=str(feature_meta.get("embeddings_file", "visual_emb.npy")),
        paths_file=str(feature_meta.get("paths_file", "clip_paths_abs.json")),
        k=int(feature_meta.get("knn_k", 50)),
        metric=str(feature_meta.get("knn_metric", "cosine")),
        mode="mean_dist",
        include_self=bool(feature_meta.get("include_self", False)),
        normalize_for_cosine=bool(feature_meta.get("normalize_for_cosine", True)),
    )
    prototype_feature = PrototypeMarginCLIPFaiss(
        cache_dir=embedding_root,
        emb_file=str(feature_meta.get("embeddings_file", "visual_emb.npy")),
        paths_file=str(feature_meta.get("paths_file", "clip_paths_abs.json")),
        centroid_file=str(feature_meta.get("centroid_file", "prototypes_k200.npy")),
        top_m=int(feature_meta.get("prototype_top_m", 8)),
        normalize=bool(feature_meta.get("normalize_for_cosine", True)),
    )

    iterator = tqdm(
        subset_records,
        desc="Extracting coverage raw features",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    extracted: List[Dict[str, object]] = []
    for record in iterator:
        image_rel = str(record["image_rel"])
        annotation_rel = str(record.get("annotation_rel", ""))
        image_path = os.path.abspath(os.path.join(subset_root, image_rel))
        meta = {"img_path": image_path, "path": image_path}

        extracted.append(
            {
                "image_rel": image_rel,
                "annotation_rel": annotation_rel,
                "knn_neighbor_distances_raw": knn_feature.get_vector_score(None, meta=meta).astype(np.float32),
                "prototype_distances_raw": prototype_feature.get_vector_score(None, meta=meta).astype(np.float32),
            }
        )
    return extracted


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract raw coverage features for a COCO-Stuff subset.")
    parser.add_argument("--subset-root", default="./data/coco_stuff50k")
    parser.add_argument("--index-path", default=None, help="Defaults to <subset-root>/sample_index.npy")
    parser.add_argument("--output-root", default="./data/data_feature/coverage")
    parser.add_argument("--embedding-root", default="./data/data_feature/coverage/visual_embedding")
    parser.add_argument("--embeddings-file", default="visual_emb.npy")
    parser.add_argument("--paths-file", default="clip_paths_abs.json")
    parser.add_argument("--centroid-file", default="prototypes_k200.npy")
    parser.add_argument("--knn-k", type=int, default=50)
    parser.add_argument("--prototype-top-m", type=int, default=8)
    parser.add_argument("--knn-metric", default="cosine")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    subset_root = os.path.abspath(args.subset_root)
    index_path = os.path.abspath(args.index_path or os.path.join(subset_root, "sample_index.npy"))
    output_root = os.path.abspath(args.output_root)
    embedding_root = os.path.abspath(args.embedding_root)
    feature_meta = {
        "embeddings_file": args.embeddings_file,
        "paths_file": args.paths_file,
        "centroid_file": args.centroid_file,
        "knn_k": int(args.knn_k),
        "knn_metric": args.knn_metric,
        "prototype_top_m": int(args.prototype_top_m),
        "normalize_for_cosine": True,
    }

    print(f"[coverage-extract] subset_root={subset_root}")
    print(f"[coverage-extract] index_path={index_path}")
    print(f"[coverage-extract] output_root={output_root}")
    print(f"[coverage-extract] embedding_root={embedding_root}")
    print(f"[coverage-extract] feature_meta={feature_meta}")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Subset index not found: {index_path}")

    subset_records = load_subset_records(index_path)
    total_records = len(subset_records)
    subset_records = limit_subset_records(subset_records, limit=args.limit)
    print(f"[coverage-extract] loaded {total_records} subset records")
    print(f"[coverage-extract] processing {len(subset_records)} subset records (limit={args.limit})")

    records = extract_coverage_records(
        subset_root=subset_root,
        subset_records=subset_records,
        embedding_root=embedding_root,
        feature_meta=feature_meta,
        show_progress=not args.no_progress,
    )
    stats = compute_coverage_global_stats(records)
    records_path, stats_path, config_path = save_coverage_feature_bundle(
        output_root=output_root,
        records=records,
        stats=stats,
        subset_root=subset_root,
        index_path=index_path,
        feature_meta=feature_meta,
    )

    print(f"[coverage-extract] saved records: {records_path}")
    print(f"[coverage-extract] saved stats: {stats_path}")
    print(f"[coverage-extract] saved config: {config_path}")
    print(f"[coverage-extract] extracted {len(records)} records")


if __name__ == "__main__":
    main()
