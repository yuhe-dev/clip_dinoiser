"""Structured analysis artifacts for completed research runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .agentic_registry import write_agentic_artifact, write_agentic_markdown
from .contracts import ExperimentCard
from .registry import load_json, resolve_repo_path
from .runtime import utc_now_iso


def _load_optional_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


def _next_hypotheses(card: ExperimentCard, judge_report: Dict[str, Any], result_bundle: Dict[str, Any]) -> List[str]:
    if card.loop_kind == "noise_floor":
        return [
            "Measure same-subset multi-seed training noise to separate optimization signal from stochasticity.",
            "Delay downstream search expansion until learner sensitivity is clearer.",
        ]
    if card.loop_kind == "same_subset_multi_seed":
        ratio = float(judge_report.get("result_summary", {}).get("noise_to_global_floor_ratio", 1.0))
        if ratio < 1.0:
            return [
                "Audit learner sensitivity under broader training regimes because composition effects may still be recoverable.",
                "Design feature intervention experiments now that training noise is bounded below the global floor.",
            ]
        return [
            "Investigate whether the current learner is too insensitive to composition changes.",
            "Tighten protocol controls before attributing deltas to data composition.",
        ]
    if card.loop_kind == "feature_intervention_matrix":
        screen_passed = bool(judge_report.get("result_summary", {}).get("screen_passed", False))
        if screen_passed:
            return [
                "Promote the strongest learner-axis cells to Tier B with shuffled and matched-random controls.",
                "Keep teacher frozen and validation full while measuring whether real features beat controls.",
            ]
        return [
            "Revisit learner adaptability scope, probe-axis choice, or materialization fidelity before escalating.",
            "Do not expand downstream slice optimization until at least one learner-axis cell exceeds its learner-specific noise floor.",
        ]
    return ["Refine the next phase-specific hypothesis based on the completed evidence."]


def build_analysis_brief(
    card: ExperimentCard,
    *,
    context_packet: Dict[str, Any] | None = None,
    result_bundle: Dict[str, Any],
    judge_report: Dict[str, Any],
    evaluation_rubric: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    context_packet = context_packet or {}
    summary = dict(result_bundle.get("summary", {}))
    reasons = list(judge_report.get("reasons", []))
    findings = [str(item) for item in reasons]
    if card.loop_kind == "same_subset_multi_seed":
        findings.append(
            "Fixed-subset training noise is now available as a reusable baseline for later Phase 1 comparisons."
        )
    if card.loop_kind == "feature_intervention_matrix":
        findings.append(
            "Learner-specific noise and feature-axis response are now summarized in a shared screening artifact."
        )
    return {
        "generated_by": "agentic_analyst_v1",
        "generated_at_utc": utc_now_iso(),
        "experiment_id": card.experiment_id,
        "phase": card.phase,
        "loop_kind": card.loop_kind,
        "status": card.status,
        "judge_decision": str(judge_report.get("decision", "")),
        "evidence_level": str(judge_report.get("evidence_level", "")),
        "key_findings": findings,
        "result_summary": summary,
        "verdict": (
            "Result is strong enough to advance the current branch."
            if str(judge_report.get("decision", "")).lower() == "promote"
            else "Result should not yet advance the branch without more evidence."
        ),
        "branch_recommendation": {
            "current_card": card.experiment_id,
            "suggested_next_hypotheses": _next_hypotheses(card, judge_report, result_bundle),
            "recommended_actions": list(judge_report.get("recommended_actions", [])),
        },
        "rubric_alignment": {
            "has_rubric": bool(evaluation_rubric),
            "primary_metric": str((evaluation_rubric or {}).get("primary_metric", "")),
            "promote_rule": str((evaluation_rubric or {}).get("promote_rule", "")),
        },
        "source_paths": {
            "context_packet": context_packet.get("source_path", ""),
        },
    }


def render_analysis_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Analysis Brief",
        "",
        f"- `experiment_id`: {payload.get('experiment_id', '')}",
        f"- `phase`: {payload.get('phase', '')}",
        f"- `loop_kind`: {payload.get('loop_kind', '')}",
        f"- `judge_decision`: {payload.get('judge_decision', '')}",
        f"- `evidence_level`: {payload.get('evidence_level', '')}",
        "",
        "## Key Findings",
        "",
    ]
    for item in payload.get("key_findings", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Verdict", "", str(payload.get("verdict", "")), "", "## Next Hypotheses", ""])
    for item in payload.get("branch_recommendation", {}).get("suggested_next_hypotheses", []):
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def write_analysis_artifacts(
    card: ExperimentCard,
    *,
    repo_root: str | Path,
    context_packet: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    output_dir = resolve_repo_path(repo_root, card.output_dir)
    result_bundle = _load_optional_json(output_dir / "result_bundle.json")
    judge_report = _load_optional_json(output_dir / "judge_report.json")
    evaluation_rubric = _load_optional_json(output_dir / "agentic" / "evaluation_rubric.json")
    if not result_bundle or not judge_report:
        return {}
    analysis = build_analysis_brief(
        card,
        context_packet=context_packet,
        result_bundle=result_bundle,
        judge_report=judge_report,
        evaluation_rubric=evaluation_rubric,
    )
    analysis_path = write_agentic_artifact(output_dir, "analysis_brief", analysis)
    write_agentic_markdown(output_dir, "analysis_brief", render_analysis_markdown(analysis))
    return {
        "analysis_brief_path": str(analysis_path.resolve()),
    }
