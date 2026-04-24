"""Noise-floor summarization for labeled random-subset experiments."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .contracts import ResultBundle


def load_jsonl_rows(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_metric_value(row: Dict[str, Any], metric_name: str) -> float | None:
    label_metrics = row.get("label_metrics", {})
    for key in ("summary", "full_summary", "proxy_summary"):
        summary = label_metrics.get(key)
        if isinstance(summary, dict) and metric_name in summary:
            value = summary[metric_name]
            if value is not None:
                return float(value)
    if metric_name in row and row[metric_name] is not None:
        return float(row[metric_name])
    return None


def _percentile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile for empty values")
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * q
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return values[lower_index]
    lower_value = values[lower_index]
    upper_value = values[upper_index]
    weight = rank - lower_index
    return lower_value + (upper_value - lower_value) * weight


def _ordered_counter(counter: Counter[Any]) -> Dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter)}


def summarize_metric_rows(rows: Iterable[Dict[str, Any]], metric_name: str) -> Dict[str, Any]:
    values: List[float] = []
    source_counter: Counter[str] = Counter()
    budget_counter: Counter[int] = Counter()
    subset_seeds = set()
    training_seeds = set()
    sample_ids: List[str] = []

    for row in rows:
        value = extract_metric_value(row, metric_name)
        if value is None:
            continue
        values.append(value)
        source_counter[str(row.get("source", "unknown"))] += 1
        budget_counter[int(row.get("budget", -1))] += 1
        if "subset_seed" in row:
            subset_seeds.add(row["subset_seed"])
        if "training_seed" in row:
            training_seeds.add(row["training_seed"])
        experiment_id = row.get("experiment_id")
        if experiment_id and len(sample_ids) < 5:
            sample_ids.append(str(experiment_id))

    if not values:
        raise ValueError(f"no labeled rows with metric '{metric_name}' were found")

    ordered_values = sorted(values)
    summary = {
        "count": len(values),
        "metric_name": metric_name,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p10": _percentile(ordered_values, 0.10),
        "p90": _percentile(ordered_values, 0.90),
        "source_counts": _ordered_counter(source_counter),
        "budget_counts": _ordered_counter(budget_counter),
        "subset_seed_count": len(subset_seeds),
        "training_seed_count": len(training_seeds),
        "sample_experiment_ids": sample_ids,
    }
    return summary


def build_noise_floor_bundle(
    experiment_id: str,
    input_path: str | Path,
    metric_name: str = "mIoU",
    metadata: Dict[str, Any] | None = None,
) -> ResultBundle:
    rows = load_jsonl_rows(input_path)
    summary = summarize_metric_rows(rows, metric_name=metric_name)
    return ResultBundle(
        experiment_id=experiment_id,
        loop_kind="noise_floor",
        input_path=str(input_path),
        metric_name=metric_name,
        summary=summary,
        sample_ids=list(summary.get("sample_experiment_ids", [])),
        metadata=dict(metadata or {}),
    )
