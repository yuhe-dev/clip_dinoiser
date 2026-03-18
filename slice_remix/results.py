from __future__ import annotations

import json
import os
from typing import Any


def load_result_entries(results_dir: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for root, _, files in os.walk(results_dir):
        for filename in files:
            if not filename.endswith("_result_entry.json"):
                continue
            path = os.path.join(root, filename)
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            candidate_id = payload.get("candidate_id")
            result_path = payload.get("result_path")
            if not candidate_id or not result_path:
                continue
            indexed[str(candidate_id)] = dict(payload)
    return indexed


def build_result_manifest_rows(
    rows: list[dict[str, Any]],
    result_entries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        execution = row.get("execution", {})
        if not isinstance(execution, dict):
            continue
        baseline_manifest_path = execution.get("baseline_manifest_path")
        if not baseline_manifest_path:
            continue
        candidate_id = str(row.get("candidate_id"))
        baseline_id = os.path.splitext(os.path.basename(str(baseline_manifest_path)))[0]
        baseline_entry = result_entries.get(baseline_id)
        candidate_entry = result_entries.get(candidate_id)
        if baseline_entry is None or candidate_entry is None:
            continue
        manifest_rows.append(
            {
                "candidate_id": candidate_id,
                "baseline_id": baseline_id,
                "baseline_result_path": baseline_entry["result_path"],
                "candidate_result_path": candidate_entry["result_path"],
                "baseline_timing": baseline_entry.get("timing"),
                "candidate_timing": candidate_entry.get("timing"),
            }
        )
    return manifest_rows
