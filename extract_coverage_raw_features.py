import argparse
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from feature_utils.data_feature.bundle import RawBundleIO, RawFeatureBundle, build_raw_feature_stats
from feature_utils.data_feature.extraction import CoverageRawExtractor


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
    return build_raw_feature_stats(records=records, feature_keys=COVERAGE_FEATURE_KEYS)


def save_coverage_feature_bundle(
    output_root: str,
    records: Sequence[Dict[str, object]],
    stats: Dict[str, object],
    subset_root: str,
    index_path: str,
    feature_meta: Dict[str, object],
) -> Tuple[str, str, str]:
    bundle = RawFeatureBundle(
        dimension_name="coverage",
        records=list(records),
        stats=stats,
        feature_config={
            "subset_root": subset_root,
            "index_path": index_path,
            "feature_meta": feature_meta,
            "records_file": "coverage_raw_features.npy",
            "stats_file": "coverage_global_stats.json",
        },
    )
    paths = RawBundleIO().save(bundle, output_root)
    return paths["records_path"], paths["stats_path"], paths["config_path"]


def extract_coverage_records(
    subset_root: str,
    subset_records: Sequence[Dict[str, object]],
    embedding_root: str,
    feature_meta: Dict[str, object],
    show_progress: bool = True,
) -> List[Dict[str, object]]:
    extractor = CoverageRawExtractor()
    extractor_feature_meta = dict(feature_meta)
    extractor_feature_meta["embedding_root"] = embedding_root
    return extractor.extract_records(
        subset_root=subset_root,
        subset_records=subset_records,
        feature_meta=extractor_feature_meta,
        show_progress=show_progress,
    )


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
