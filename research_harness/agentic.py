"""Agentic orchestration scaffold on top of the execution harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .analyst import write_analysis_artifacts
from .agentic_judge import apply_agentic_judge, write_judgment_artifacts
from .context_packet import build_context_packet
from .contracts import JudgeReport, ResultBundle
from .contracts import ExperimentCard
from .literature import write_literature_artifacts
from .planner import write_planning_artifacts
from .registry import load_json, resolve_repo_path


def build_context_with_source(
    *,
    repo_root: str | Path,
    card: ExperimentCard,
    card_path: str | Path,
) -> Dict[str, Any]:
    payload = build_context_packet(
        repo_root=repo_root,
        card=card,
        card_path=card_path,
    )
    payload["source_path"] = str((resolve_repo_path(repo_root, card.output_dir) / "agentic" / "context_snapshot.json").resolve())
    return payload


def ensure_agentic_artifacts(
    *,
    repo_root: str | Path,
    card: ExperimentCard,
    card_path: str | Path,
    execute_literature_search: bool = False,
) -> Dict[str, Any]:
    output_dir = resolve_repo_path(repo_root, card.output_dir) if card.output_dir else Path(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    context_packet = build_context_with_source(repo_root=repo_root, card=card, card_path=card_path)
    from .agentic_registry import write_agentic_artifact

    context_path = write_agentic_artifact(output_dir, "context_snapshot", context_packet)
    context_packet["source_path"] = str(context_path.resolve())

    paths: Dict[str, Any] = {
        "context_snapshot_path": str(context_path.resolve()),
    }
    paths.update(write_planning_artifacts(card, repo_root=repo_root, context_packet=context_packet))
    paths.update(write_analysis_artifacts(card, repo_root=repo_root, context_packet=context_packet))
    result_bundle_path = output_dir / "result_bundle.json"
    judge_report_path = output_dir / "judge_report.json"
    rubric_path = output_dir / "agentic" / "evaluation_rubric.json"
    if result_bundle_path.exists() and judge_report_path.exists() and rubric_path.exists():
        result_payload = load_json(result_bundle_path)
        report_payload = load_json(judge_report_path)
        rubric_payload = load_json(rubric_path)
        final_report, judgment_brief = apply_agentic_judge(
            card=card,
            context_packet=context_packet,
            bundle=ResultBundle(
                experiment_id=str(result_payload.get("experiment_id", card.experiment_id)),
                loop_kind=str(result_payload.get("loop_kind", card.loop_kind)),
                input_path=str(result_payload.get("input_path", card.input_path)),
                metric_name=str(result_payload.get("metric_name", card.metric_name)),
                summary=dict(result_payload.get("summary", {})),
                sample_ids=list(result_payload.get("sample_ids", [])),
                metadata=dict(result_payload.get("metadata", {})),
            ),
            mechanical_report=JudgeReport(
                experiment_id=str(report_payload.get("experiment_id", card.experiment_id)),
                decision=str(report_payload.get("decision", "")),
                evidence_level=str(report_payload.get("evidence_level", "")),
                result_summary=dict(report_payload.get("result_summary", {})),
                reasons=[str(item) for item in report_payload.get("reasons", [])],
                recommended_actions=[str(item) for item in report_payload.get("recommended_actions", [])],
                requires_literature_radar=bool(report_payload.get("requires_literature_radar", False)),
                protocol_contamination=bool(report_payload.get("protocol_contamination", False)),
            ),
            evaluation_rubric=rubric_payload,
        )
        _ = final_report
        paths.update(write_judgment_artifacts(output_dir, judgment_brief))

    should_run_radar = card.loop_kind == "literature_radar" or bool(card.metadata.get("proposal_origin") == "dynamic_literature_radar")
    if should_run_radar:
        analysis_payload = {}
        analysis_path = output_dir / "agentic" / "analysis_brief.json"
        if analysis_path.exists():
            analysis_payload = load_json(analysis_path)
        paths.update(
            write_literature_artifacts(
                card,
                repo_root=repo_root,
                context_packet=context_packet,
                analysis_brief=analysis_payload,
                execute_search=execute_literature_search,
            )
        )
    return paths
