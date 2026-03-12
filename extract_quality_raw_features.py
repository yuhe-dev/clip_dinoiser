import argparse
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from feature_utils.data_feature.bundle import RawBundleIO, RawFeatureBundle, build_raw_feature_stats
from feature_utils.data_feature.extraction import QualityRawExtractor


QUALITY_FEATURE_KEYS: Tuple[str, ...] = ("laplacian_raw", "noise_pca_raw", "bga_raw")


def load_subset_records(index_path: str) -> List[Dict[str, object]]:
    records = np.load(index_path, allow_pickle=True)
    return [dict(item) for item in records.tolist()]


def compute_global_stats(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return build_raw_feature_stats(records=records, feature_keys=QUALITY_FEATURE_KEYS)


def save_quality_feature_bundle(
    output_root: str,
    records: Sequence[Dict[str, object]],
    stats: Dict[str, object],
    subset_root: str,
    index_path: str,
    feature_meta: Dict[str, object],
) -> Tuple[str, str, str]:
    bundle = RawFeatureBundle(
        dimension_name="quality",
        records=list(records),
        stats=stats,
        feature_config={
            "subset_root": subset_root,
            "index_path": index_path,
            "feature_meta": feature_meta,
            "records_file": "quality_raw_features.npy",
            "stats_file": "quality_global_stats.json",
        },
    )
    paths = RawBundleIO().save(bundle, output_root)
    return paths["records_path"], paths["stats_path"], paths["config_path"]


def extract_quality_records(
    subset_root: str,
    subset_records: Sequence[Dict[str, object]],
    feature_meta: Dict[str, object],
    show_progress: bool = True,
) -> List[Dict[str, object]]:
    extractor = QualityRawExtractor()
    return extractor.extract_records(
        subset_root=subset_root,
        subset_records=subset_records,
        feature_meta=feature_meta,
        show_progress=show_progress,
    )


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
