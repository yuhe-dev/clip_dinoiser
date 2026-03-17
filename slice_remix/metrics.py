from __future__ import annotations

import json
from typing import Any


def load_experiment_metrics(result_path: str) -> dict[str, Any]:
    with open(result_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not payload:
        raise ValueError("experiment result payload is empty")

    task_name = next(iter(payload))
    task_payload = payload[task_name]
    return {
        "task": task_name,
        "summary": dict(task_payload.get("summary", {})),
        "per_class": dict(task_payload.get("per_class", {})),
        "raw": payload,
    }


def extract_metric_value(payload: dict[str, Any], metric_path: str) -> float:
    current: Any = payload
    for token in metric_path.split("."):
        if not isinstance(current, dict) or token not in current:
            raise KeyError(f"metric path '{metric_path}' not found")
        current = current[token]
    return float(current)
