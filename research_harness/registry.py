"""Persistence helpers for experiment cards, bundles, judge reports, and manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .contracts import ExperimentCard, JudgeReport, ResultBundle, RunManifest


def repo_root_from_path(path: str | Path) -> Path:
    return Path(path).resolve().parent


def resolve_repo_path(repo_root: str | Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(repo_root).resolve() / path


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_experiment_card(path: str | Path) -> ExperimentCard:
    return ExperimentCard.from_dict(load_json(path))


def write_experiment_card(path: str | Path, card: ExperimentCard) -> Path:
    target = Path(path)
    write_json(target, card.to_dict())
    return target


def load_judge_policy(path: str | Path) -> Dict[str, Any]:
    return load_json(path)


def write_result_bundle(output_dir: str | Path, bundle: ResultBundle) -> Path:
    target = Path(output_dir) / "result_bundle.json"
    write_json(target, bundle.to_dict())
    return target


def write_judge_report_json(output_dir: str | Path, report: JudgeReport) -> Path:
    target = Path(output_dir) / "judge_report.json"
    write_json(target, report.to_dict())
    return target


def render_judge_report_markdown(report: JudgeReport) -> str:
    lines = [
        "# Judge Report",
        "",
        "## Basic Info",
        "",
        f"- Experiment ID: {report.experiment_id}",
        f"- Decision: {report.decision}",
        f"- Evidence Level: {report.evidence_level}",
        "",
        "## Result Summary",
        "",
    ]
    for key, value in sorted(report.result_summary.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Reasons", ""])
    for reason in report.reasons:
        lines.append(f"- {reason}")
    lines.extend(["", "## Recommended Actions", ""])
    for action in report.recommended_actions:
        lines.append(f"- {action}")
    lines.extend(
        [
            "",
            "## Flags",
            "",
            f"- protocol_contamination: {report.protocol_contamination}",
            f"- requires_literature_radar: {report.requires_literature_radar}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_judge_report_markdown(output_dir: str | Path, report: JudgeReport) -> Path:
    target = Path(output_dir) / "judge_report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_judge_report_markdown(report), encoding="utf-8")
    return target


def write_run_manifest(output_dir: str | Path, manifest: RunManifest) -> Path:
    target = Path(output_dir) / "run_manifest.json"
    write_json(target, manifest.to_dict())
    return target
