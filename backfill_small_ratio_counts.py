import argparse
import json
import os
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from extract_difficulty_raw_features import compute_global_stats
from feature_utils.data_feature.implementations.difficulty import SmallObjectRatioCOCOStuff


def load_raw_records(records_path: str) -> List[Dict[str, object]]:
    records = np.load(records_path, allow_pickle=True)
    return [dict(item) for item in records.tolist()]


def load_feature_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_feature_config(config_path: str, config: Dict[str, object]) -> None:
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def _load_mask(annotation_path: str) -> np.ndarray:
    with Image.open(annotation_path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def backfill_small_ratio_counts_for_records(
    records: Sequence[Dict[str, object]],
    subset_root: str,
    thresholds: Optional[Sequence[float]] = None,
    ignore_index: int = 255,
    use_things_only: bool = False,
    verify_profile: bool = False,
    progress_interval: int = 100,
    log_fn: Callable[[str], None] = print,
) -> List[Dict[str, object]]:
    metric = SmallObjectRatioCOCOStuff(
        thresholds=list(thresholds) if thresholds is not None else None,
        default_ignore_index=int(ignore_index),
        use_things_only=bool(use_things_only),
    )
    updated: List[Dict[str, object]] = []
    total_records = int(len(records))
    log_fn(f"[small-ratio-backfill] starting count backfill for {total_records} records")
    for idx, record in enumerate(records, start=1):
        new_record = dict(record)
        annotation_rel = str(record["annotation_rel"])
        annotation_path = os.path.join(subset_root, annotation_rel)
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation not found for backfill: {annotation_path}")
        mask = _load_mask(annotation_path)
        profile, count = metric.get_profile_and_count(
            image=np.zeros(mask.shape + (3,), dtype=np.uint8),
            mask=mask,
            meta={
                "ignore_index": int(ignore_index),
                "use_things_only": bool(use_things_only),
            },
        )
        if verify_profile and "small_ratio_raw" in new_record:
            existing_profile = np.asarray(new_record["small_ratio_raw"], dtype=np.float32)
            if existing_profile.shape == profile.shape and not np.allclose(existing_profile, profile, atol=1e-6):
                raise ValueError(
                    "Existing small_ratio_raw does not match recomputed profile for "
                    f"{annotation_rel}; aborting backfill."
                )
        new_record["small_ratio_num_values"] = int(count)
        updated.append(new_record)
        if progress_interval > 0 and (idx == total_records or idx % progress_interval == 0):
            log_fn(f"[small-ratio-backfill] processed {idx}/{total_records} records")
    return updated


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill real small_ratio connected-component counts into difficulty raw features.")
    parser.add_argument("--records-path", required=True, help="Path to difficulty_raw_features.npy")
    parser.add_argument("--config-path", required=True, help="Path to difficulty_feature_config.json")
    parser.add_argument("--subset-root", default=None, help="Overrides subset_root from the feature config if provided.")
    parser.add_argument("--output-records-path", default=None, help="Defaults to overwriting --records-path")
    parser.add_argument("--output-stats-path", default=None, help="Defaults to sibling difficulty_global_stats.json next to output records.")
    parser.add_argument("--update-config", action="store_true", help="Update the config file to point at the new records/stats filenames.")
    parser.add_argument("--verify-profile", action="store_true", help="Abort if recomputed small_ratio profiles differ from the stored raw profile.")
    parser.add_argument("--progress-interval", type=int, default=100, help="Print progress every N records. Use 0 to disable periodic progress logs.")
    return parser


def main() -> None:
    started_at = time.time()
    args = build_argparser().parse_args()
    records_path = os.path.abspath(args.records_path)
    config_path = os.path.abspath(args.config_path)
    config = load_feature_config(config_path)
    subset_root = os.path.abspath(args.subset_root or str(config["subset_root"]))
    feature_meta = dict(config.get("feature_meta", {}))
    thresholds = feature_meta.get("small_ratio_thresholds")
    ignore_index = int(feature_meta.get("ignore_index", 255))
    use_things_only = bool(feature_meta.get("use_things_only", False))
    print(f"[small-ratio-backfill] records_path={records_path}")
    print(f"[small-ratio-backfill] config_path={config_path}")
    print(f"[small-ratio-backfill] subset_root={subset_root}")

    print("[small-ratio-backfill] loading raw records")
    records = load_raw_records(records_path)
    print(f"[small-ratio-backfill] loaded {len(records)} raw records")
    updated_records = backfill_small_ratio_counts_for_records(
        records=records,
        subset_root=subset_root,
        thresholds=thresholds,
        ignore_index=ignore_index,
        use_things_only=use_things_only,
        verify_profile=bool(args.verify_profile),
        progress_interval=max(int(args.progress_interval), 0),
    )

    output_records_path = os.path.abspath(args.output_records_path or records_path)
    if args.output_stats_path:
        output_stats_path = os.path.abspath(args.output_stats_path)
    else:
        output_stats_path = os.path.join(os.path.dirname(output_records_path), "difficulty_global_stats.json")

    os.makedirs(os.path.dirname(output_records_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_stats_path), exist_ok=True)
    print(f"[small-ratio-backfill] saving updated records to {output_records_path}")
    np.save(output_records_path, np.asarray(updated_records, dtype=object), allow_pickle=True)

    print("[small-ratio-backfill] recomputing global stats")
    stats = compute_global_stats(updated_records)
    with open(output_stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    if args.update_config:
        config["records_file"] = os.path.basename(output_records_path)
        config["stats_file"] = os.path.basename(output_stats_path)
        save_feature_config(config_path, config)

    print(f"[small-ratio-backfill] records={len(updated_records)}")
    print(f"[small-ratio-backfill] output_records_path={output_records_path}")
    print(f"[small-ratio-backfill] output_stats_path={output_stats_path}")
    print(f"[small-ratio-backfill] elapsed_sec={time.time() - started_at:.2f}")


if __name__ == "__main__":
    main()
