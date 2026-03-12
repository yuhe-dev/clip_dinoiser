import argparse
import json
import math
import os
import re
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np


def load_schema(schema_path: str) -> Dict[str, object]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw_records(records_path: str) -> List[Dict[str, object]]:
    records = np.load(records_path, allow_pickle=True)
    return [dict(item) for item in records.tolist()]


def apply_value_transform(values: np.ndarray, transform_name: str) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)

    name = (transform_name or "identity").lower().strip()
    vals = np.asarray(values, dtype=np.float32)
    if name == "identity":
        return vals
    if name == "log1p":
        return np.log1p(np.maximum(vals, 0.0)).astype(np.float32)
    if name == "identity_clamp_nonnegative":
        return np.maximum(vals, 0.0).astype(np.float32)
    raise ValueError(f"Unsupported value_transform='{transform_name}'")


def fit_distribution_bin_edges(raw_arrays: Sequence[np.ndarray], feature_spec: Dict[str, object]) -> np.ndarray:
    transformed = [
        apply_value_transform(np.asarray(arr, dtype=np.float32), str(feature_spec.get("value_transform", "identity")))
        for arr in raw_arrays
        if np.asarray(arr).size > 0
    ]
    num_bins = int(feature_spec["num_bins"])
    range_mode = str(feature_spec["range_mode"])
    range_params = dict(feature_spec.get("range_params", {}))

    if range_mode == "fixed":
        low = float(range_params["min"])
        high = float(range_params["max"])
    elif range_mode == "robust_global":
        if not transformed:
            low, high = 0.0, 1.0
        else:
            merged = np.concatenate(transformed, axis=0)
            low = float(np.quantile(merged, float(range_params.get("lower_quantile", 0.01))))
            high = float(np.quantile(merged, float(range_params.get("upper_quantile", 0.99))))
    else:
        raise ValueError(f"Unsupported range_mode='{range_mode}'")

    if not np.isfinite(low) or not np.isfinite(high):
        low, high = 0.0, 1.0
    if high <= low:
        high = low + 1.0
    return np.linspace(low, high, num_bins + 1, dtype=np.float32)


def _parse_bin_range(description: str) -> range:
    match = re.search(r"bins?\s+(\d+)-(\d+)", description)
    if not match:
        raise ValueError(f"Cannot parse bin range from '{description}'")
    start, end = int(match.group(1)), int(match.group(2))
    return range(start, end + 1)


def _safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def _distribution_summary(values: np.ndarray, hist: np.ndarray, summary_fields: Dict[str, str]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for key, description in summary_fields.items():
        if key == "mean":
            summary[key] = float(values.mean()) if values.size else 0.0
        elif key == "std":
            summary[key] = float(values.std()) if values.size else 0.0
        elif key.startswith("q"):
            q = float(key[1:]) / 100.0
            summary[key] = _safe_quantile(values, q)
        elif "histogram bins" in description:
            bins = _parse_bin_range(description)
            summary[key] = float(hist[list(bins)].sum()) if hist.size else 0.0
        else:
            summary[key] = 0.0
    return summary


def encode_distribution_feature(raw_values: np.ndarray, feature_spec: Dict[str, object], bin_edges: np.ndarray) -> Dict[str, object]:
    values = apply_value_transform(np.asarray(raw_values, dtype=np.float32), str(feature_spec.get("value_transform", "identity")))
    num_values = int(values.size)
    empty_flag = int(num_values == 0)
    num_bins = int(feature_spec["num_bins"])

    if num_values == 0:
        hist = np.zeros((num_bins,), dtype=np.float32)
    else:
        clipped = np.clip(values, float(bin_edges[0]), float(bin_edges[-1]))
        counts, _ = np.histogram(clipped, bins=bin_edges)
        hist = counts.astype(np.float32)
        total = float(hist.sum())
        if total > 0:
            hist /= total

    summary = _distribution_summary(values, hist, dict(feature_spec.get("summary_fields", {})))
    return {
        "encoding": "distribution",
        "value_transform": str(feature_spec.get("value_transform", "identity")),
        "empty_flag": empty_flag,
        "num_values": num_values,
        "log_num_values": float(math.log1p(num_values)),
        "hist": hist,
        "summary": summary,
        "model_input_fields": list(feature_spec.get("model_input_fields", [])),
    }


def _profile_summary(profile: np.ndarray, delta_profile: np.ndarray, summary_fields: Dict[str, str]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for key, description in summary_fields.items():
        if key == "mean":
            summary[key] = float(profile.mean()) if profile.size else 0.0
        elif key == "std":
            summary[key] = float(profile.std()) if profile.size else 0.0
        elif key.startswith("q"):
            q = float(key[1:]) / 100.0
            summary[key] = _safe_quantile(profile, q)
        elif key == "first_active_bin":
            nz = np.where(delta_profile > 0)[0]
            summary[key] = float(nz[0] / max(len(delta_profile) - 1, 1)) if nz.size else 0.0
        elif key in {"nearest_distance", "nearest_prototype_distance"}:
            summary[key] = float(profile[0]) if profile.size else 0.0
        elif key == "farthest_distance":
            summary[key] = float(profile[-1]) if profile.size else 0.0
        elif key == "density_score":
            summary[key] = float(1.0 / (1e-12 + float(profile.mean()))) if profile.size else 0.0
        elif key == "prototype_margin_top2":
            summary[key] = float(profile[1] - profile[0]) if profile.size >= 2 else 0.0
        elif key == "prototype_margin_top5":
            summary[key] = float(profile[4] - profile[0]) if profile.size >= 5 else 0.0
        elif "delta_profile bins" in description:
            bins = _parse_bin_range(description)
            valid = [idx for idx in bins if idx < len(delta_profile)]
            summary[key] = float(delta_profile[valid].sum()) if valid else 0.0
        else:
            summary[key] = 0.0
    return summary


def encode_profile_feature(
    raw_values: np.ndarray,
    feature_spec: Dict[str, object],
    feature_name: str = "",
    source_num_values: Optional[int] = None,
) -> Dict[str, object]:
    profile = apply_value_transform(np.asarray(raw_values, dtype=np.float32), str(feature_spec.get("value_transform", "identity")))
    profile = np.asarray(profile, dtype=np.float32)
    if profile.size == 0:
        delta_profile = np.asarray([], dtype=np.float32)
    else:
        delta_profile = np.empty_like(profile)
        delta_profile[0] = profile[0]
        if profile.size > 1:
            delta_profile[1:] = np.maximum(profile[1:] - profile[:-1], 0.0)

    num_values = int(source_num_values) if source_num_values is not None else int(profile.size)
    summary = _profile_summary(profile, delta_profile, dict(feature_spec.get("summary_fields", {})))
    return {
        "encoding": "profile",
        "value_transform": str(feature_spec.get("value_transform", "identity")),
        "empty_flag": int(num_values == 0),
        "num_values": num_values,
        "log_num_values": float(math.log1p(num_values)),
        "profile": profile,
        "delta_profile": delta_profile,
        "summary": summary,
        "model_input_fields": list(feature_spec.get("model_input_fields", [])),
    }


def process_dimension_records(
    raw_records: Sequence[Dict[str, object]],
    dimension_schema: Dict[str, object],
    dimension_name: str = "",
    progress_interval: int = 100,
    log_fn: Callable[[str], None] = print,
) -> List[Dict[str, object]]:
    feature_specs = dict(dimension_schema.get("features", {}))
    bin_edges_by_feature: Dict[str, np.ndarray] = {}
    label = dimension_name or str(dimension_schema.get("schema_version", "dimension"))
    total_records = int(len(raw_records))
    log_fn(f"[postprocess] {label}: preparing {total_records} raw records")

    for feature_name, feature_spec in feature_specs.items():
        spec = dict(feature_spec)
        if spec["encoding"] != "distribution":
            continue
        raw_key = str(spec["raw_key"])
        log_fn(f"[postprocess] {label}: fitting bin edges for {feature_name}")
        arrays = [np.asarray(record.get(raw_key, np.asarray([], dtype=np.float32)), dtype=np.float32) for record in raw_records]
        bin_edges_by_feature[feature_name] = fit_distribution_bin_edges(arrays, spec)

    processed: List[Dict[str, object]] = []
    for idx, record in enumerate(raw_records, start=1):
        feature_blocks: Dict[str, object] = {}
        for feature_name, feature_spec in feature_specs.items():
            spec = dict(feature_spec)
            raw = np.asarray(record.get(spec["raw_key"], np.asarray([], dtype=np.float32)), dtype=np.float32)
            if spec["encoding"] == "distribution":
                feature_blocks[feature_name] = encode_distribution_feature(raw, spec, bin_edges_by_feature[feature_name])
            elif spec["encoding"] == "profile":
                source_count_key = spec.get("source_count_key")
                source_num_values = None
                if source_count_key:
                    raw_count = record.get(str(source_count_key))
                    if raw_count is not None:
                        source_num_values = int(raw_count)
                feature_blocks[feature_name] = encode_profile_feature(
                    raw,
                    spec,
                    feature_name=feature_name,
                    source_num_values=source_num_values,
                )
            else:
                raise ValueError(f"Unsupported encoding='{spec['encoding']}'")

        processed.append(
            {
                "image_rel": record.get("image_rel", ""),
                "annotation_rel": record.get("annotation_rel", ""),
                "schema_version": str(dimension_schema["schema_version"]),
                "features": feature_blocks,
            }
        )
        if progress_interval > 0 and (idx == total_records or idx % progress_interval == 0):
            log_fn(f"[postprocess] {label}: processed {idx}/{total_records} records")
    return processed


def compute_processed_summary(processed_records: Sequence[Dict[str, object]]) -> Dict[str, object]:
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


def save_processed_bundle(
    output_root: str,
    dimension_name: str,
    processed_records: Sequence[Dict[str, object]],
    dimension_schema: Dict[str, object],
    source_records_path: str,
    source_stats_path: str,
    source_config_path: str,
    schema_source_path: str,
) -> Dict[str, str]:
    os.makedirs(output_root, exist_ok=True)
    records_path = os.path.join(output_root, f"{dimension_name}_processed_features.npy")
    schema_path = os.path.join(output_root, f"{dimension_name}_processed_schema.json")
    config_path = os.path.join(output_root, f"{dimension_name}_processing_config.json")
    summary_path = os.path.join(output_root, f"{dimension_name}_processed_summary.json")

    np.save(records_path, np.asarray(list(processed_records), dtype=object), allow_pickle=True)
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(dimension_schema, f, indent=2, ensure_ascii=False)

    config = {
        "dimension": dimension_name,
        "schema_version": dimension_schema["schema_version"],
        "source_records_path": source_records_path,
        "source_stats_path": source_stats_path,
        "source_config_path": source_config_path,
        "schema_source_path": schema_source_path,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    summary = compute_processed_summary(processed_records)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
        "records_path": records_path,
        "schema_path": schema_path,
        "config_path": config_path,
        "summary_path": summary_path,
    }


def process_dimension_bundle(
    dimension_name: str,
    data_root: str,
    schema_path: str,
    skip_missing: bool = False,
    progress_interval: int = 100,
) -> Dict[str, str]:
    schema = load_schema(schema_path)
    dimension_schema = dict(schema["dimensions"][dimension_name])
    output_root = os.path.join(data_root, dimension_name)
    raw_records_path = os.path.join(output_root, f"{dimension_name}_raw_features.npy")
    raw_stats_path = os.path.join(output_root, f"{dimension_name}_global_stats.json")
    raw_config_path = os.path.join(output_root, f"{dimension_name}_feature_config.json")

    if not os.path.exists(raw_records_path):
        if skip_missing:
            return {}
        raise FileNotFoundError(f"Raw records not found for dimension '{dimension_name}': {raw_records_path}")

    raw_records = load_raw_records(raw_records_path)
    print(f"[postprocess] {dimension_name}: loaded {len(raw_records)} raw records from {raw_records_path}")
    processed_records = process_dimension_records(
        raw_records,
        dimension_schema,
        dimension_name=dimension_name,
        progress_interval=progress_interval,
    )
    print(f"[postprocess] {dimension_name}: saving processed bundle")
    return save_processed_bundle(
        output_root=output_root,
        dimension_name=dimension_name,
        processed_records=processed_records,
        dimension_schema=dimension_schema,
        source_records_path=raw_records_path,
        source_stats_path=raw_stats_path,
        source_config_path=raw_config_path,
        schema_source_path=schema_path,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Postprocess raw feature bundles into processed feature bundles.")
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=["quality", "difficulty", "coverage"],
        choices=["quality", "difficulty", "coverage"],
    )
    parser.add_argument("--data-root", default="./data/data_feature")
    parser.add_argument("--schema-path", default="./docs/feature_schema/unified_processed_feature_schema.json")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=100, help="Print progress every N records per dimension. Use 0 to disable periodic progress logs.")
    return parser


def main() -> None:
    started_at = time.time()
    args = build_argparser().parse_args()
    print(f"[postprocess] data_root={os.path.abspath(args.data_root)}")
    print(f"[postprocess] schema_path={os.path.abspath(args.schema_path)}")
    print(f"[postprocess] dimensions={args.dimensions}")
    for dimension_name in args.dimensions:
        result = process_dimension_bundle(
            dimension_name=dimension_name,
            data_root=os.path.abspath(args.data_root),
            schema_path=os.path.abspath(args.schema_path),
            skip_missing=bool(args.skip_missing),
            progress_interval=max(int(args.progress_interval), 0),
        )
        if result:
            print(f"[postprocess] {dimension_name}: wrote {result['records_path']}")
        else:
            print(f"[postprocess] {dimension_name}: skipped missing raw records")
    print(f"[postprocess] elapsed_sec={time.time() - started_at:.2f}")


if __name__ == "__main__":
    main()
