"""Context-aware overlay judge that aligns fixed metrics with frozen rubrics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Tuple

from .agentic_registry import write_agentic_artifact, write_agentic_markdown
from .contracts import ExperimentCard, JudgeReport, ResultBundle
from .runtime import utc_now_iso


def _noise_floor_alignment(summary: Dict[str, Any], thresholds: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    count = int(summary.get("count", 0))
    stdev = float(summary.get("stdev", 0.0))
    observed_range = float(summary.get("range", 0.0))
    minimum_labeled_runs = int(thresholds.get("minimum_labeled_runs", 30))
    narrow_std_threshold = float(thresholds.get("narrow_std_threshold", 0.05))
    narrow_range_threshold = float(thresholds.get("narrow_range_threshold", 0.20))
    aligned = (
        count >= minimum_labeled_runs
        and stdev <= narrow_std_threshold
        and observed_range <= narrow_range_threshold
    )
    return aligned, {
        "count": count,
        "stdev": stdev,
        "range": observed_range,
        "minimum_labeled_runs": minimum_labeled_runs,
        "narrow_std_threshold": narrow_std_threshold,
        "narrow_range_threshold": narrow_range_threshold,
    }


def _same_subset_alignment(summary: Dict[str, Any], thresholds: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    completed = int(summary.get("completed_seed_count", 0))
    ratio = float(summary.get("noise_to_global_floor_ratio", 1.0))
    minimum_completed_runs = int(thresholds.get("minimum_completed_runs", 3))
    comparable_noise_ratio = float(thresholds.get("comparable_noise_ratio", 1.0))
    aligned = completed >= minimum_completed_runs and ratio < comparable_noise_ratio
    return aligned, {
        "completed_seed_count": completed,
        "noise_to_global_floor_ratio": ratio,
        "minimum_completed_runs": minimum_completed_runs,
        "comparable_noise_ratio": comparable_noise_ratio,
    }


def _literature_alignment(summary: Dict[str, Any], thresholds: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    ranked_result_count = int(summary.get("ranked_result_count", 0))
    reproduce_count = int(summary.get("reproduce_count", 0))
    search_error_ratio = float(summary.get("search_error_ratio", 0.0))
    minimum_ranked_results = int(thresholds.get("minimum_ranked_results", 6))
    minimum_reproduce_candidates = int(thresholds.get("minimum_reproduce_candidates", 1))
    maximum_search_error_ratio = float(thresholds.get("maximum_search_error_ratio", 0.5))
    aligned = (
        ranked_result_count >= minimum_ranked_results
        and reproduce_count >= minimum_reproduce_candidates
        and search_error_ratio <= maximum_search_error_ratio
    )
    return aligned, {
        "ranked_result_count": ranked_result_count,
        "reproduce_count": reproduce_count,
        "search_error_ratio": search_error_ratio,
        "minimum_ranked_results": minimum_ranked_results,
        "minimum_reproduce_candidates": minimum_reproduce_candidates,
        "maximum_search_error_ratio": maximum_search_error_ratio,
    }


def _learner_ladder_alignment(summary: Dict[str, Any], thresholds: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    completed = int(summary.get("completed_regime_count", 0))
    regime_range = float(summary.get("regime_range", 0.0))
    minimum_completed_regimes = int(thresholds.get("minimum_completed_regimes", 3))
    training_noise_reference = float(thresholds.get("training_noise_reference", 0.0089442719))
    meaningful_sensitivity_multiplier = float(thresholds.get("meaningful_sensitivity_multiplier", 1.5))
    threshold = training_noise_reference * meaningful_sensitivity_multiplier
    aligned = completed >= minimum_completed_regimes and regime_range > threshold
    return aligned, {
        "completed_regime_count": completed,
        "regime_range": regime_range,
        "minimum_completed_regimes": minimum_completed_regimes,
        "training_noise_reference": training_noise_reference,
        "meaningful_sensitivity_multiplier": meaningful_sensitivity_multiplier,
        "meaningful_sensitivity_threshold": threshold,
    }


def _learner_adaptability_alignment(summary: Dict[str, Any], thresholds: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    real_axes_with_signal_count = int(summary.get("real_axes_with_signal_count", 0))
    best_ratio = float(summary.get("best_real_response_to_noise_ratio", 0.0))
    directional_consistency = float(summary.get("minimum_directional_consistency", 0.0))
    real_beats_shuffled = bool(summary.get("real_beats_shuffled", False))
    real_beats_random = bool(summary.get("real_beats_random", False))
    full_validation = bool(summary.get("full_validation", False))
    teacher_frozen = bool(summary.get("teacher_frozen", False))
    minimum_real_axes_with_signal = int(thresholds.get("minimum_real_axes_with_signal", 1))
    minimum_ratio = float(thresholds.get("tier_b.minimum_response_to_noise_ratio", 2.0))
    minimum_consistency = float(thresholds.get("tier_b.minimum_directional_consistency", 0.67))
    require_real_beats_shuffled = bool(thresholds.get("tier_b.require_real_beats_shuffled", True))
    require_real_beats_random = bool(thresholds.get("tier_b.require_real_beats_random", True))

    aligned = (
        real_axes_with_signal_count >= minimum_real_axes_with_signal
        and best_ratio >= minimum_ratio
        and directional_consistency >= minimum_consistency
        and full_validation
        and teacher_frozen
        and (not require_real_beats_shuffled or real_beats_shuffled)
        and (not require_real_beats_random or real_beats_random)
    )
    return aligned, {
        "real_axes_with_signal_count": real_axes_with_signal_count,
        "best_real_response_to_noise_ratio": best_ratio,
        "minimum_directional_consistency": directional_consistency,
        "real_beats_shuffled": real_beats_shuffled,
        "real_beats_random": real_beats_random,
        "full_validation": full_validation,
        "teacher_frozen": teacher_frozen,
        "minimum_real_axes_with_signal": minimum_real_axes_with_signal,
        "threshold.minimum_response_to_noise_ratio": minimum_ratio,
        "threshold.minimum_directional_consistency": minimum_consistency,
    }


def evaluate_rubric_alignment(
    bundle: ResultBundle,
    rubric: Dict[str, Any] | None,
    *,
    report_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    rubric = dict(rubric or {})
    contract = dict(rubric.get("judge_contract", {}))
    contract_type = str(contract.get("contract_type", "generic_design_review"))
    thresholds = dict(contract.get("thresholds", {}))
    merged_summary = dict(bundle.summary)
    merged_summary.update(dict(report_summary or {}))
    if contract_type == "noise_floor":
        aligned, details = _noise_floor_alignment(merged_summary, thresholds)
    elif contract_type == "same_subset_multi_seed":
        aligned, details = _same_subset_alignment(merged_summary, thresholds)
    elif contract_type == "literature_radar":
        aligned, details = _literature_alignment(merged_summary, thresholds)
    elif contract_type == "learner_sensitivity_ladder":
        aligned, details = _learner_ladder_alignment(merged_summary, thresholds)
    elif contract_type == "learner_adaptability_audit":
        aligned, details = _learner_adaptability_alignment(merged_summary, thresholds)
    else:
        aligned = True
        details = {"note": "No machine-checked contract; kept mechanical decision."}
    return {
        "contract_type": contract_type,
        "aligned": aligned,
        "details": details,
    }


def build_judgment_brief(
    *,
    card: ExperimentCard,
    context_packet: Dict[str, Any] | None,
    bundle: ResultBundle,
    mechanical_report: JudgeReport,
    evaluation_rubric: Dict[str, Any] | None,
) -> Dict[str, Any]:
    context_packet = dict(context_packet or {})
    alignment = evaluate_rubric_alignment(
        bundle,
        evaluation_rubric,
        report_summary=mechanical_report.result_summary,
    )
    final_decision = str(mechanical_report.decision)
    reasons = list(mechanical_report.reasons)
    reasons.append(
        f"Rubric contract `{alignment['contract_type']}` alignment={alignment['aligned']}."
    )
    task_snapshot = dict(context_packet.get("task_snapshot", {}))
    runtime_snapshot = dict(context_packet.get("runtime_snapshot", {}))
    if task_snapshot.get("research_state"):
        reasons.append(f"Context research_state={task_snapshot.get('research_state')}.")
    if runtime_snapshot.get("judge_decision"):
        reasons.append(f"Prior runtime snapshot decision={runtime_snapshot.get('judge_decision')}.")

    if str(mechanical_report.decision) == "promote" and not bool(alignment.get("aligned", True)):
        final_decision = "park"
        reasons.append("Downgraded from mechanical promote because the frozen rubric contract is not yet satisfied.")
    elif str(mechanical_report.decision) in {"park", "rerun"} and bool(alignment.get("aligned", False)):
        final_decision = "promote"
        reasons.append("Upgraded because the result satisfies the frozen rubric contract despite a conservative mechanical judgment.")

    recommended_actions = list(mechanical_report.recommended_actions)
    if final_decision == "park":
        recommended_actions.append("Revisit the design pack or execution recipe before escalating this branch.")
    elif final_decision == "promote":
        recommended_actions.append("Carry this result forward as a context-aware promoted artifact for the current branch.")

    return {
        "generated_by": "agentic_judge_v1",
        "generated_at_utc": utc_now_iso(),
        "experiment_id": card.experiment_id,
        "phase": card.phase,
        "loop_kind": card.loop_kind,
        "mechanical_decision": str(mechanical_report.decision),
        "final_decision": final_decision,
        "contract_type": alignment["contract_type"],
        "contract_alignment": alignment["details"],
        "alignment_passed": bool(alignment["aligned"]),
        "reasons": reasons,
        "recommended_actions": recommended_actions,
        "context_signals": {
            "research_state": str(task_snapshot.get("research_state", "")),
            "next_action": str(task_snapshot.get("next_action", "")),
            "runtime_status": str(runtime_snapshot.get("runtime_status", "")),
            "depends_on": list(card.depends_on),
        },
    }


def apply_agentic_judge(
    *,
    card: ExperimentCard,
    context_packet: Dict[str, Any] | None,
    bundle: ResultBundle,
    mechanical_report: JudgeReport,
    evaluation_rubric: Dict[str, Any] | None,
) -> Tuple[JudgeReport, Dict[str, Any]]:
    judgment_brief = build_judgment_brief(
        card=card,
        context_packet=context_packet,
        bundle=bundle,
        mechanical_report=mechanical_report,
        evaluation_rubric=evaluation_rubric,
    )
    final_report = replace(
        mechanical_report,
        decision=str(judgment_brief["final_decision"]),
        reasons=list(judgment_brief["reasons"]),
        recommended_actions=list(judgment_brief["recommended_actions"]),
    )
    return final_report, judgment_brief


def render_judgment_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Judgment Brief",
        "",
        f"- `experiment_id`: {payload.get('experiment_id', '')}",
        f"- `loop_kind`: {payload.get('loop_kind', '')}",
        f"- `mechanical_decision`: {payload.get('mechanical_decision', '')}",
        f"- `final_decision`: {payload.get('final_decision', '')}",
        f"- `contract_type`: {payload.get('contract_type', '')}",
        f"- `alignment_passed`: {payload.get('alignment_passed', False)}",
        "",
        "## Reasons",
        "",
    ]
    for item in payload.get("reasons", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Recommended Actions", ""])
    for item in payload.get("recommended_actions", []):
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def write_judgment_artifacts(output_dir: str | Path, judgment_brief: Dict[str, Any]) -> Dict[str, str]:
    output_dir = Path(output_dir)
    brief_path = write_agentic_artifact(output_dir, "judgment_brief", judgment_brief)
    markdown_path = write_agentic_markdown(output_dir, "judgment_brief", render_judgment_markdown(judgment_brief))
    return {
        "judgment_brief_path": str(brief_path.resolve()),
        "judgment_markdown_path": str(markdown_path.resolve()),
    }
