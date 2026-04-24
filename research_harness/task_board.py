"""Aggregate task-level progress across experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .registry import load_experiment_card, load_json
from .task_states import validate_task_plan
from .runtime import utc_now_iso


def _task_plan_path(card_output_dir: str | Path) -> Path:
    return Path(card_output_dir) / "task_plan.json"


def build_task_board(repo_root: str | Path, scan_dir: str | Path) -> Dict[str, Any]:
    repo_root = Path(repo_root)
    scan_dir = Path(scan_dir)
    entries: List[Dict[str, Any]] = []
    for card_path in sorted(scan_dir.glob("*.json")):
        card = load_experiment_card(card_path)
        output_dir = (repo_root / card.output_dir).resolve()
        plan_path = _task_plan_path(output_dir)
        payload = {}
        task_plan_valid = False
        task_plan_error = "missing_task_plan"
        if plan_path.exists():
            payload = load_json(plan_path)
            task_plan_valid, task_plan_error = validate_task_plan(payload)
        entries.append(
            {
                "experiment_id": card.experiment_id,
                "status": card.status,
                "phase": card.phase,
                "loop_kind": card.loop_kind,
                "output_dir": str(output_dir),
                "task_plan_path": str(plan_path) if plan_path.exists() else "",
                "current_step": payload.get("current_step", ""),
                "next_action": payload.get("next_action", ""),
                "research_state": payload.get("research_state", ""),
                "acceptance_status": payload.get("acceptance_status", ""),
                "task_plan_valid": task_plan_valid,
                "task_plan_error": task_plan_error,
            }
        )
    return {
        "generated_at_utc": utc_now_iso(),
        "entry_count": len(entries),
        "entries": entries,
    }
