import math
import re
from typing import Dict, Optional, Sequence

import numpy as np


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
    raise ValueError("Unsupported value_transform='%s'" % transform_name)


def _parse_bin_range(description: str) -> range:
    match = re.search(r"bins?\s+(\d+)-(\d+)", description)
    if not match:
        raise ValueError("Cannot parse bin range from '%s'" % description)
    start, end = int(match.group(1)), int(match.group(2))
    return range(start, end + 1)


def _safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


class DistributionFeatureEncoder:
    def __init__(self, feature_spec: Dict[str, object]):
        self.feature_spec = dict(feature_spec)
        self.bin_edges: Optional[np.ndarray] = None

    def fit(self, raw_arrays: Sequence[np.ndarray]) -> None:
        transformed = [
            apply_value_transform(np.asarray(arr, dtype=np.float32), str(self.feature_spec.get("value_transform", "identity")))
            for arr in raw_arrays
            if np.asarray(arr).size > 0
        ]
        num_bins = int(self.feature_spec["num_bins"])
        range_mode = str(self.feature_spec["range_mode"])
        range_params = dict(self.feature_spec.get("range_params", {}))
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
            raise ValueError("Unsupported range_mode='%s'" % range_mode)
        if not np.isfinite(low) or not np.isfinite(high):
            low, high = 0.0, 1.0
        if high <= low:
            high = low + 1.0
        self.bin_edges = np.linspace(low, high, num_bins + 1, dtype=np.float32)

    def _build_summary(self, values: np.ndarray, hist: np.ndarray) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        for key, description in dict(self.feature_spec.get("summary_fields", {})).items():
            if key == "mean":
                summary[key] = float(values.mean()) if values.size else 0.0
            elif key == "std":
                summary[key] = float(values.std()) if values.size else 0.0
            elif key.startswith("q"):
                summary[key] = _safe_quantile(values, float(key[1:]) / 100.0)
            elif "histogram bins" in description:
                bins = _parse_bin_range(description)
                summary[key] = float(hist[list(bins)].sum()) if hist.size else 0.0
            else:
                summary[key] = 0.0
        return summary

    def transform(self, raw_values: np.ndarray, record: Dict[str, object]) -> Dict[str, object]:
        del record
        if self.bin_edges is None:
            raise RuntimeError("DistributionFeatureEncoder.fit() must be called before transform().")
        values = apply_value_transform(np.asarray(raw_values, dtype=np.float32), str(self.feature_spec.get("value_transform", "identity")))
        num_values = int(values.size)
        num_bins = int(self.feature_spec["num_bins"])
        if num_values == 0:
            hist = np.zeros((num_bins,), dtype=np.float32)
        else:
            clipped = np.clip(values, float(self.bin_edges[0]), float(self.bin_edges[-1]))
            counts, _ = np.histogram(clipped, bins=self.bin_edges)
            hist = counts.astype(np.float32)
            if float(hist.sum()) > 0:
                hist /= float(hist.sum())
        return {
            "encoding": "distribution",
            "value_transform": str(self.feature_spec.get("value_transform", "identity")),
            "empty_flag": int(num_values == 0),
            "num_values": num_values,
            "log_num_values": float(math.log1p(num_values)),
            "hist": hist,
            "summary": self._build_summary(values, hist),
            "model_input_fields": list(self.feature_spec.get("model_input_fields", [])),
        }


class ProfileFeatureEncoder:
    def __init__(self, feature_spec: Dict[str, object]):
        self.feature_spec = dict(feature_spec)

    def _build_summary(self, profile: np.ndarray, delta_profile: np.ndarray) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        for key, description in dict(self.feature_spec.get("summary_fields", {})).items():
            if key == "mean":
                summary[key] = float(profile.mean()) if profile.size else 0.0
            elif key == "std":
                summary[key] = float(profile.std()) if profile.size else 0.0
            elif key.startswith("q"):
                summary[key] = _safe_quantile(profile, float(key[1:]) / 100.0)
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

    def transform(self, raw_values: np.ndarray, record: Dict[str, object]) -> Dict[str, object]:
        profile = apply_value_transform(np.asarray(raw_values, dtype=np.float32), str(self.feature_spec.get("value_transform", "identity")))
        profile = np.asarray(profile, dtype=np.float32)
        if profile.size == 0:
            delta_profile = np.asarray([], dtype=np.float32)
        else:
            delta_profile = np.empty_like(profile)
            delta_profile[0] = profile[0]
            if profile.size > 1:
                delta_profile[1:] = np.maximum(profile[1:] - profile[:-1], 0.0)
        source_num_values = None
        source_count_key = self.feature_spec.get("source_count_key")
        if source_count_key:
            raw_count = record.get(str(source_count_key))
            if raw_count is not None:
                source_num_values = int(raw_count)
        num_values = int(source_num_values) if source_num_values is not None else int(profile.size)
        return {
            "encoding": "profile",
            "value_transform": str(self.feature_spec.get("value_transform", "identity")),
            "empty_flag": int(num_values == 0),
            "num_values": num_values,
            "log_num_values": float(math.log1p(num_values)),
            "profile": profile,
            "delta_profile": delta_profile,
            "summary": self._build_summary(profile, delta_profile),
            "model_input_fields": list(self.feature_spec.get("model_input_fields", [])),
        }
