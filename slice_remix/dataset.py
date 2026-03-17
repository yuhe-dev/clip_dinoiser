from __future__ import annotations

import json
from typing import Any

import numpy as np


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def build_response_row(
    *,
    baseline_trial_id: str,
    candidate_id: str,
    baseline_mixture: list[float] | np.ndarray,
    target_mixture: list[float] | np.ndarray,
    delta_q: list[float] | np.ndarray,
    delta_phi: dict[str, list[float] | np.ndarray],
    context: dict[str, Any],
    measured_gain: float | None,
) -> dict[str, Any]:
    return {
        "baseline_trial_id": baseline_trial_id,
        "candidate_id": candidate_id,
        "baseline_mixture": _to_jsonable(baseline_mixture),
        "target_mixture": _to_jsonable(target_mixture),
        "delta_q": _to_jsonable(delta_q),
        "delta_phi": _to_jsonable(delta_phi),
        "context": _to_jsonable(context),
        "measured_gain": _to_jsonable(measured_gain),
    }


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
