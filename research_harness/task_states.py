"""Research-native task state machine helpers."""

from __future__ import annotations

from typing import Dict, Tuple

RESEARCH_STATES = [
    "hypothesis",
    "design",
    "audit",
    "execution",
    "verification",
    "judgment",
    "acceptance",
]

RESEARCH_STATE_SET = set(RESEARCH_STATES)


def validate_task_plan(plan: Dict[str, object]) -> Tuple[bool, str]:
    if not isinstance(plan, dict):
        return False, "task_plan is not a dict"
    state = str(plan.get("research_state", "")).strip().lower()
    next_state = str(plan.get("next_state", "")).strip().lower()
    if not state:
        return False, "task_plan missing research_state"
    if state not in RESEARCH_STATE_SET:
        return False, f"unknown research_state '{state}'"
    if next_state and next_state not in RESEARCH_STATE_SET:
        return False, f"unknown next_state '{next_state}'"
    return True, ""
