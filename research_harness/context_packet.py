"""Context packet generation for long-running research sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from .contracts import ExperimentCard
from .registry import load_json, resolve_repo_path, write_json
from .runtime import utc_now_iso


DEFAULT_MEMORY_PATHS = [
    "AGENTS.md",
    ".slicetune/MEMORY.md",
    ".slicetune/context/program.md",
    ".slicetune/context/playbook.md",
    ".slicetune/state/board.md",
    ".slicetune/state/decision_log.md",
]


def _load_optional_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


def _tail_lines(path: Path, limit: int = 40) -> List[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[-limit:]


def build_context_packet(
    *,
    repo_root: str | Path,
    card: ExperimentCard,
    card_path: str | Path,
    memory_paths: Iterable[str] | None = None,
) -> Dict[str, Any]:
    root = Path(repo_root).resolve()
    memory_values = list(memory_paths or [])
    if not memory_values:
        memory_values = list(DEFAULT_MEMORY_PATHS)

    output_dir = resolve_repo_path(root, card.output_dir) if card.output_dir else None
    task_plan = _load_optional_json(output_dir / "task_plan.json") if output_dir else {}
    judge_report = _load_optional_json(output_dir / "judge_report.json") if output_dir else {}
    result_bundle = _load_optional_json(output_dir / "result_bundle.json") if output_dir else {}
    runtime_index = _load_optional_json(root / ".slicetune/state/runtime_index.json")
    task_board = _load_optional_json(root / ".slicetune/state/task_board.json")
    proposal_index = _load_optional_json(root / ".slicetune/state/proposal_index.json")

    runtime_entry = {}
    for entry in runtime_index.get("entries", []):
        if str(entry.get("experiment_id", "")) == card.experiment_id:
            runtime_entry = dict(entry)
            break

    task_entry = {}
    for entry in task_board.get("entries", []):
        if str(entry.get("experiment_id", "")) == card.experiment_id:
            task_entry = dict(entry)
            break

    related_cards = [str(item) for item in card.depends_on]
    if card.experiment_id in {str(item.get("experiment_id", "")) for item in proposal_index.get("proposals", [])}:
        related_cards.append(card.experiment_id)

    memory_index = []
    for raw in memory_values:
        resolved = resolve_repo_path(root, raw)
        memory_index.append(
            {
                "path": str(resolved),
                "exists": resolved.exists(),
            }
        )

    board_tail = _tail_lines(root / ".slicetune/state/board.md", limit=30)
    decision_tail = _tail_lines(root / ".slicetune/state/decision_log.md", limit=30)

    return {
        "generated_at_utc": utc_now_iso(),
        "experiment_id": card.experiment_id,
        "card_path": str(Path(card_path).resolve()),
        "loop_kind": card.loop_kind,
        "phase": card.phase,
        "status": card.status,
        "owner": card.owner,
        "hypothesis": card.hypothesis,
        "budget_tier": card.budget_tier,
        "depends_on": related_cards,
        "memory_index": memory_index,
        "task_snapshot": {
            "research_state": str(task_plan.get("research_state", task_entry.get("research_state", ""))),
            "next_action": str(task_plan.get("next_action", task_entry.get("next_action", ""))),
            "acceptance_status": str(task_plan.get("acceptance_status", task_entry.get("acceptance_status", ""))),
            "recent_facts": list(task_plan.get("recent_facts", []))[:8],
            "blockers": list(task_plan.get("blockers", []))[:8],
        },
        "runtime_snapshot": {
            "attempt_count": int(runtime_entry.get("attempt_count", card.attempt_count)),
            "judge_decision": str(runtime_entry.get("judge_decision", judge_report.get("decision", ""))),
            "runtime_status": str(runtime_entry.get("status", card.status)),
        },
        "artifact_snapshot": {
            "output_dir": str(output_dir.resolve()) if output_dir else "",
            "has_task_plan": bool(task_plan),
            "has_result_bundle": bool(result_bundle),
            "has_judge_report": bool(judge_report),
            "debate_bundle_path": card.debate_bundle_path,
        },
        "repo_memory_tail": {
            "board_md_tail": board_tail,
            "decision_log_tail": decision_tail,
        },
    }


def write_context_packet(
    session_dir: str | Path,
    *,
    repo_root: str | Path,
    card: ExperimentCard,
    card_path: str | Path,
    memory_paths: Iterable[str] | None = None,
) -> Path:
    payload = build_context_packet(
        repo_root=repo_root,
        card=card,
        card_path=card_path,
        memory_paths=memory_paths,
    )
    target = Path(session_dir) / "context_packet.json"
    write_json(target, payload)
    return target
