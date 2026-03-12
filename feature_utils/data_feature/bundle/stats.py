from typing import Dict, Sequence, Tuple

import numpy as np


def build_raw_feature_stats(
    records: Sequence[Dict[str, object]],
    feature_keys: Tuple[str, ...],
) -> Dict[str, object]:
    stats: Dict[str, object] = {
        "num_samples": int(len(records)),
        "features": {},
    }
    for feature_key in feature_keys:
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


def build_processed_feature_summary(processed_records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {"num_samples": int(len(processed_records)), "features": {}}
    if not processed_records:
        return summary

    feature_names = list(processed_records[0]["features"].keys())
    for feature_name in feature_names:
        empty_flags = [int(record["features"][feature_name]["empty_flag"]) for record in processed_records]
        num_values = [int(record["features"][feature_name]["num_values"]) for record in processed_records]
        summary["features"][feature_name] = {
            "empty_samples": int(sum(empty_flags)),
            "num_values_min": int(min(num_values)) if num_values else 0,
            "num_values_max": int(max(num_values)) if num_values else 0,
            "num_values_mean": float(np.mean(num_values)) if num_values else 0.0,
        }
    return summary
