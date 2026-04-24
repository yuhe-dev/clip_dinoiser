"""Research-native taskflow reconciliation for experiment cards."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from .contracts import ExperimentCard
from .controller import load_controller_policy, transition_card_status, validate_debate_gate, validate_phase_gate
from .loop_catalog import card_is_execution_ready, execution_readiness_reason
from .registry import load_experiment_card, load_json, resolve_repo_path
from .runtime import utc_now_iso
from .task_progress import write_progress_artifacts
from .task_states import validate_task_plan


def _existing_valid_task_plan(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    payload = load_json(path)
    ok, _reason = validate_task_plan(payload)
    return payload if ok else None


def _task_plan_matches_card_status(plan: Dict[str, Any], card: ExperimentCard) -> bool:
    status = str(card.status)
    research_state = str(plan.get("research_state", ""))
    acceptance_status = str(plan.get("acceptance_status", ""))
    if status == "completed" and acceptance_status == "awaiting_human_review":
        return False
    if status == "completed" and research_state == "acceptance":
        return False
    if status == "awaiting_human_review" and acceptance_status != "awaiting_human_review":
        return False
    if status == "planned" and research_state not in {"hypothesis", "design", ""}:
        return False
    return True


def _artifact_flags(repo_root: str | Path, card: ExperimentCard) -> Dict[str, bool]:
    root = Path(repo_root)
    output_dir = resolve_repo_path(root, card.output_dir) if card.output_dir else Path("")
    flags = {
        "debate_bundle_exists": False,
        "result_bundle_exists": False,
        "judge_report_exists": False,
        "run_manifest_exists": False,
    }
    if card.debate_bundle_path:
        flags["debate_bundle_exists"] = resolve_repo_path(root, card.debate_bundle_path).exists()
    if card.output_dir:
        flags["result_bundle_exists"] = (output_dir / "result_bundle.json").exists()
        flags["judge_report_exists"] = (output_dir / "judge_report.json").exists()
        flags["run_manifest_exists"] = (output_dir / "run_manifest.json").exists()
    return flags


def _infer_state(card: ExperimentCard, artifacts: Dict[str, bool]) -> Tuple[str, str]:
    status = str(card.status)
    if status == "planned":
        return "hypothesis", "design"
    if status == "blocked_debate_gate":
        return "design", "audit"
    if status in {"queued", "stale_requeued", "blocked_phase_gate", "blocked_preflight", "blocked_retry_limit", "failed_execution"}:
        if card.requires_debate and not artifacts["debate_bundle_exists"]:
            return "design", "audit"
        return "audit", "execution"
    if status in {"claimed", "running"}:
        return "execution", "verification"
    if status == "awaiting_human_review":
        return "acceptance", ""
    if status == "completed":
        if artifacts["judge_report_exists"]:
            return "judgment", "acceptance" if card.human_review_required or card.phase_completion_candidate else ""
        if artifacts["result_bundle_exists"] or artifacts["run_manifest_exists"]:
            return "verification", "judgment"
        return "execution", "verification"
    return "audit", "execution"


def _infer_next_action(card: ExperimentCard, research_state: str, artifacts: Dict[str, bool]) -> str:
    status = str(card.status)
    if research_state == "hypothesis":
        return "author_hypothesis_and_scope"
    if research_state == "design":
        return "prepare_or_revise_debate_bundle"
    if research_state == "audit":
        if not card_is_execution_ready(card):
            return "author_execution_recipe_or_runtime_handler"
        if status == "blocked_preflight":
            return "repair_runtime_and_requeue"
        if status == "blocked_phase_gate":
            return "align_phase_or_policy_then_requeue"
        if status == "failed_execution":
            return "inspect_attempt_and_requeue"
        return "satisfy_gates_then_queue"
    if research_state == "execution":
        return "monitor_or_resume_execution"
    if research_state == "verification":
        return "verify_artifacts_and_metrics"
    if research_state == "judgment":
        return "review_judge_report_and_choose_next_branch"
    if research_state == "acceptance":
        return "wait_human_acceptance"
    return "inspect_card"


def _generic_tasks(card: ExperimentCard, research_state: str, next_state: str, artifacts: Dict[str, bool]) -> List[Dict[str, Any]]:
    execution_ready = card_is_execution_ready(card)
    tasks: List[Dict[str, Any]] = [
        {
            "task_id": "frame_hypothesis",
            "title": "Frame research hypothesis",
            "stage": "hypothesis",
            "status": "completed" if research_state not in {"hypothesis"} else "running",
            "detail": card.hypothesis or "hypothesis pending refinement",
        },
        {
            "task_id": "design_protocol",
            "title": "Design experiment or debate protocol",
            "stage": "design",
            "status": "completed" if artifacts["debate_bundle_exists"] or research_state not in {"hypothesis", "design"} else ("running" if research_state == "design" else "pending"),
            "detail": "debate bundle ready" if artifacts["debate_bundle_exists"] else "debate/design artifact pending",
        },
        {
            "task_id": "audit_readiness",
            "title": "Audit readiness before execution",
            "stage": "audit",
            "status": "completed" if research_state in {"execution", "verification", "judgment", "acceptance"} else ("running" if research_state == "audit" else "pending"),
            "detail": f"card_status={card.status}",
        },
        {
            "task_id": "run_execution",
            "title": "Execute experiment loop",
            "stage": "execution",
            "status": (
                "blocked"
                if not execution_ready and research_state in {"audit", "execution"}
                else "completed"
                if research_state in {"verification", "judgment", "acceptance"}
                else ("running" if research_state == "execution" else "pending")
            ),
            "detail": execution_readiness_reason(card) if not execution_ready else ("result bundle present" if artifacts["result_bundle_exists"] else "execution not finished"),
        },
        {
            "task_id": "verify_outputs",
            "title": "Verify outputs and summarize metrics",
            "stage": "verification",
            "status": "completed" if research_state in {"judgment", "acceptance"} else ("running" if research_state == "verification" else "pending"),
            "detail": "judge artifacts present" if artifacts["judge_report_exists"] else "verification pending",
        },
        {
            "task_id": "judge_branch",
            "title": "Judge branch outcome",
            "stage": "judgment",
            "status": "completed" if research_state == "acceptance" else ("running" if research_state == "judgment" else "pending"),
            "detail": f"next_state={next_state or 'done'}",
        },
        {
            "task_id": "human_acceptance",
            "title": "Human acceptance gate",
            "stage": "acceptance",
            "status": "running" if research_state == "acceptance" else "pending",
            "detail": "await human review" if research_state == "acceptance" else "not yet at acceptance gate",
        },
    ]
    return tasks


def build_generic_task_plan(card: ExperimentCard, *, repo_root: str | Path) -> Dict[str, Any]:
    artifacts = _artifact_flags(repo_root, card)
    research_state, next_state = _infer_state(card, artifacts)
    next_action = _infer_next_action(card, research_state, artifacts)
    acceptance_status = "not_required"
    if card.status == "completed":
        acceptance_status = "not_required"
    elif research_state == "acceptance":
        acceptance_status = "awaiting_human_review"
    elif card.human_review_required or card.phase_completion_candidate:
        acceptance_status = "pending_review"
    blockers: List[str] = []
    if research_state in {"audit", "execution"} and not card_is_execution_ready(card):
        blockers.append(execution_readiness_reason(card))
    return {
        "generated_by": "taskflow",
        "generated_at_utc": utc_now_iso(),
        "experiment_id": card.experiment_id,
        "loop_kind": card.loop_kind,
        "current_step": f"status::{card.status}",
        "next_action": next_action,
        "research_state": research_state,
        "next_state": next_state,
        "acceptance_status": acceptance_status,
        "state_history": [
            {
                "at_utc": utc_now_iso(),
                "research_state": research_state,
                "card_status": card.status,
            }
        ],
        "tasks": _generic_tasks(card, research_state, next_state, artifacts),
        "recent_facts": [
            f"card_status={card.status}",
            f"debate_bundle_exists={artifacts['debate_bundle_exists']}",
            f"result_bundle_exists={artifacts['result_bundle_exists']}",
            f"judge_report_exists={artifacts['judge_report_exists']}",
        ],
        "blockers": blockers,
    }


def reconcile_task_plan(card_path: str | Path, *, repo_root: str | Path) -> Dict[str, Any]:
    card_path = Path(card_path)
    card = load_experiment_card(card_path)
    output_dir = resolve_repo_path(repo_root, card.output_dir) if card.output_dir else Path("")
    task_plan_path = output_dir / "task_plan.json" if card.output_dir else Path("")
    existing = _existing_valid_task_plan(task_plan_path) if card.output_dir else None
    if (
        existing is not None
        and str(existing.get("generated_by", "")) != "taskflow"
        and _task_plan_matches_card_status(existing, card)
    ):
        return existing
    payload = build_generic_task_plan(card, repo_root=repo_root)
    if card.output_dir:
        write_progress_artifacts(output_dir, payload)
    return payload


def auto_advance_card(
    card_path: str | Path,
    *,
    repo_root: str | Path,
    controller_policy_path: str | Path,
) -> str:
    card_path = Path(card_path)
    card = load_experiment_card(card_path)
    policy = load_controller_policy(controller_policy_path)

    if card.status not in {"planned", "blocked_debate_gate", "blocked_phase_gate"}:
        return card.status

    phase_ok, _phase_reason = validate_phase_gate(card, policy)
    debate_ok, _debate_reason = validate_debate_gate(card, repo_root, policy)

    if card.status == "planned":
        if phase_ok and (not card.requires_debate or debate_ok) and card_is_execution_ready(card):
            transition_card_status(
                card_path,
                card,
                new_status="queued",
                reason="taskflow auto-advance after hypothesis/design gates satisfied",
                session_id="taskflow-auto-advance",
            )
            return "queued"
        return card.status

    if card.status in {"blocked_debate_gate", "blocked_phase_gate"}:
        if phase_ok and (not card.requires_debate or debate_ok) and card_is_execution_ready(card):
            transition_card_status(
                card_path,
                card,
                new_status="queued",
                reason="taskflow auto-requeue after blocked gate resolved",
                session_id="taskflow-auto-advance",
            )
            return "queued"
    return card.status


def reconcile_task_plans(scan_dir: str | Path, *, repo_root: str | Path) -> Dict[str, Any]:
    root = Path(scan_dir)
    plans: Dict[str, Any] = {}
    for card_path in sorted(root.glob("*.json")):
        plan = reconcile_task_plan(card_path, repo_root=repo_root)
        card = load_experiment_card(card_path)
        plans[card.experiment_id] = {
            "research_state": plan.get("research_state", ""),
            "next_action": plan.get("next_action", ""),
            "current_step": plan.get("current_step", ""),
        }
    return {
        "generated_at_utc": utc_now_iso(),
        "plan_count": len(plans),
        "plans": plans,
    }
