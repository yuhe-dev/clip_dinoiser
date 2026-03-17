from __future__ import annotations

from typing import Any

from .metrics import extract_metric_value


def attach_measured_gain(
    row: dict[str, Any],
    *,
    baseline_result_path: str,
    candidate_result_path: str,
    metric_path: str,
) -> dict[str, Any]:
    import json

    with open(baseline_result_path, "r", encoding="utf-8") as f:
        baseline_payload = json.load(f)
    with open(candidate_result_path, "r", encoding="utf-8") as f:
        candidate_payload = json.load(f)

    baseline_metric_value = extract_metric_value(baseline_payload, metric_path)
    candidate_metric_value = extract_metric_value(candidate_payload, metric_path)

    labeled = dict(row)
    labeled["metric_path"] = metric_path
    labeled["baseline_result_path"] = baseline_result_path
    labeled["candidate_result_path"] = candidate_result_path
    labeled["baseline_metric_value"] = baseline_metric_value
    labeled["candidate_metric_value"] = candidate_metric_value
    labeled["measured_gain"] = candidate_metric_value - baseline_metric_value
    return labeled
