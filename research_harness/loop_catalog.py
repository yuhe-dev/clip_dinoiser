"""Loop catalog for execution readiness and conductor dispatch."""

from __future__ import annotations

from .contracts import ExperimentCard


EXECUTABLE_LOOP_KINDS = {
    "noise_floor",
    "same_subset_multi_seed",
    "learner_sensitivity_ladder",
    "feature_intervention_matrix",
    "literature_radar",
}


def is_executable_loop_kind(loop_kind: str) -> bool:
    return str(loop_kind).strip() in EXECUTABLE_LOOP_KINDS


def card_is_execution_ready(card: ExperimentCard) -> bool:
    if bool(card.metadata.get("design_only", False)):
        return False
    return is_executable_loop_kind(card.loop_kind)


def execution_readiness_reason(card: ExperimentCard) -> str:
    if bool(card.metadata.get("design_only", False)):
        return "card is marked design_only and requires an execution recipe before queueing"
    if not is_executable_loop_kind(card.loop_kind):
        return f"loop_kind '{card.loop_kind}' does not have a runtime handler yet"
    return ""
