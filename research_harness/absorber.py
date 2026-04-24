"""Aggregate machine-readable runtime state from cards and artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .registry import load_json, resolve_repo_path, write_json
from .runtime import utc_now_iso
from .scheduler import list_experiment_cards


def _safe_load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


def build_runtime_index(repo_root: str | Path, scan_dir: str | Path) -> Dict[str, Any]:
    cards = list_experiment_cards(scan_dir)
    entries = []
    status_counts: Dict[str, int] = {}
    decision_counts: Dict[str, int] = {}
    for card_path, card in cards:
        output_dir = resolve_repo_path(repo_root, card.output_dir) if card.output_dir else Path("")
        result_bundle = _safe_load(output_dir / "result_bundle.json") if card.output_dir else {}
        judge_report = _safe_load(output_dir / "judge_report.json") if card.output_dir else {}
        run_manifest = _safe_load(output_dir / "run_manifest.json") if card.output_dir else {}
        attempt_manifest = {}
        if card.output_dir and card.last_attempt_id:
            attempt_path = output_dir / "attempts" / card.last_attempt_id / "attempt_manifest.json"
            attempt_manifest = _safe_load(attempt_path)

        entry = {
            "experiment_id": card.experiment_id,
            "status": card.status,
            "phase": card.phase,
            "loop_kind": card.loop_kind,
            "priority": card.priority,
            "attempt_count": card.attempt_count,
            "max_attempts": card.max_attempts,
            "last_attempt_id": card.last_attempt_id,
            "card_path": str(Path(card_path).resolve()),
            "output_dir": str(output_dir.resolve()) if card.output_dir else "",
            "judge_decision": str(judge_report.get("decision", "")),
            "evidence_level": str(judge_report.get("evidence_level", "")),
            "recommended_actions": list(judge_report.get("recommended_actions", [])),
            "result_summary": dict(judge_report.get("result_summary", {})),
            "run_manifest_path": str((output_dir / "run_manifest.json").resolve()) if card.output_dir and (output_dir / "run_manifest.json").exists() else "",
            "judge_report_path": str((output_dir / "judge_report.json").resolve()) if card.output_dir and (output_dir / "judge_report.json").exists() else "",
            "result_bundle_path": str((output_dir / "result_bundle.json").resolve()) if card.output_dir and (output_dir / "result_bundle.json").exists() else "",
            "last_attempt_status": str(attempt_manifest.get("status", "")),
            "last_attempt_finished_at_utc": str(attempt_manifest.get("finished_at_utc", "")),
            "git_sha": str(run_manifest.get("git_sha", "")),
        }
        entries.append(entry)
        status_counts[card.status] = status_counts.get(card.status, 0) + 1
        decision = entry["judge_decision"]
        if decision:
            decision_counts[decision] = decision_counts.get(decision, 0) + 1

    return {
        "generated_at_utc": utc_now_iso(),
        "card_count": len(entries),
        "status_counts": status_counts,
        "decision_counts": decision_counts,
        "entries": sorted(entries, key=lambda item: (item["phase"], item["priority"], item["experiment_id"])),
    }


def write_runtime_index(path: str | Path, payload: Dict[str, Any]) -> Path:
    write_json(path, payload)
    return Path(path)
