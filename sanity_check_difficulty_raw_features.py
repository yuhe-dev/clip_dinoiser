import argparse
import json
import os
from typing import Dict, List, Sequence

import numpy as np


DIFFICULTY_FEATURE_KEYS = ("small_ratio_raw", "visual_semantic_gap_raw", "empirical_iou_raw")


def load_difficulty_records(records_path: str) -> List[Dict[str, object]]:
    records = np.load(records_path, allow_pickle=True)
    return [dict(item) for item in records.tolist()]


def compute_feature_summary(arrays: Sequence[np.ndarray], sample_limit: int = 3) -> Dict[str, object]:
    values = [np.asarray(arr, dtype=np.float32).reshape(-1) for arr in arrays]
    lengths = [int(arr.size) for arr in values]
    non_empty = [arr for arr in values if arr.size > 0]
    merged = np.concatenate(non_empty, axis=0) if non_empty else np.asarray([], dtype=np.float32)

    summary = {
        "num_samples": int(len(values)),
        "empty_samples": int(sum(1 for length in lengths if length == 0)),
        "length": {
            "min": int(min(lengths)) if lengths else 0,
            "max": int(max(lengths)) if lengths else 0,
            "mean": float(np.mean(lengths)) if lengths else 0.0,
        },
        "values": {
            "min": float(merged.min()) if merged.size else None,
            "max": float(merged.max()) if merged.size else None,
            "mean": float(merged.mean()) if merged.size else None,
            "std": float(merged.std()) if merged.size else None,
        },
        "sample_values": [arr[:10].astype(np.float32).tolist() for arr in values[:sample_limit]],
    }
    return summary


def compute_difficulty_bundle_summary(records: Sequence[Dict[str, object]], sample_limit: int = 3) -> Dict[str, object]:
    summary: Dict[str, object] = {"num_samples": int(len(records)), "features": {}}
    for feature_key in DIFFICULTY_FEATURE_KEYS:
        arrays = [np.asarray(record.get(feature_key, np.asarray([], dtype=np.float32)), dtype=np.float32) for record in records]
        summary["features"][feature_key] = compute_feature_summary(arrays, sample_limit=sample_limit)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize extracted raw difficulty features.")
    parser.add_argument("--records-path", default="./data/data_feature/difficulty/difficulty_raw_features.npy")
    parser.add_argument("--out-json", default="./data/data_feature/difficulty/difficulty_raw_summary.json")
    parser.add_argument("--sample-limit", type=int, default=3)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    records_path = os.path.abspath(args.records_path)
    out_json = os.path.abspath(args.out_json)

    print(f"[difficulty-sanity] records_path={records_path}")
    if not os.path.exists(records_path):
        raise FileNotFoundError(f"Raw feature file not found: {records_path}")

    records = load_difficulty_records(records_path)
    print(f"[difficulty-sanity] loaded {len(records)} records")

    summary = compute_difficulty_bundle_summary(records, sample_limit=args.sample_limit)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[difficulty-sanity] saved summary: {out_json}")
    for feature_key, feature_summary in summary["features"].items():
        print(f"[difficulty-sanity] {feature_key}: {feature_summary}")


if __name__ == "__main__":
    main()
