"""Code-enforced controller logic for the long-running research harness."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from .contracts import ExperimentCard
from .debate import load_debate_bundle, validate_debate_bundle
from .registry import (
    load_experiment_card,
    load_json,
    resolve_repo_path,
    write_experiment_card,
    write_json,
)
from .runtime import add_seconds_iso, is_utc_iso_stale, utc_now_iso


STATUS_TRANSITIONS: dict[str, set[str]] = {
    "planned": {"queued", "blocked_phase_gate", "blocked_debate_gate"},
    "queued": {"claimed", "blocked_phase_gate", "blocked_debate_gate", "blocked_preflight", "blocked_retry_limit"},
    "stale_requeued": {"claimed", "blocked_phase_gate", "blocked_debate_gate", "blocked_preflight", "blocked_retry_limit"},
    "blocked_phase_gate": {"queued", "claimed"},
    "blocked_debate_gate": {"queued", "claimed"},
    "blocked_preflight": {"queued", "claimed"},
    "blocked_retry_limit": {"queued", "claimed"},
    "claimed": {"running", "stale_requeued", "failed_execution"},
    "running": {"awaiting_human_review", "completed", "failed_execution", "stale_requeued"},
    "awaiting_human_review": {"completed", "queued"},
    "failed_execution": {"queued", "claimed"},
    "completed": set(),
}

RUNNING_LIKE_STATUSES = {"claimed", "running"}


def discover_repo_root(card_path: str | Path) -> Path:
    resolved = Path(card_path).resolve()
    for candidate in [resolved.parent, *resolved.parents]:
        if (candidate / ".slicetune").is_dir() and (candidate / "AGENTS.md").exists():
            return candidate
    return resolved.parent


def load_controller_policy(path: str | Path) -> Dict[str, Any]:
    policy = load_json(path)
    policy.setdefault("allowed_phases", [])
    policy.setdefault("always_load_paths", [])
    policy.setdefault("min_debate_rounds", 2)
    policy.setdefault("require_debate_artifacts", False)
    policy.setdefault("human_review_required_for_phase_completion", True)
    policy.setdefault("session_output_dir", "artifacts/research_harness/sessions")
    policy.setdefault("lease_dir", "artifacts/research_harness/leases")
    policy.setdefault("runtime_profiles_path", ".slicetune/runtime/runtime_profiles.json")
    policy.setdefault("default_runtime_profile", "")
    policy.setdefault("default_runtime_profile_candidates", [])
    policy.setdefault("preflight", {})
    policy.setdefault("heartbeat_interval_seconds", 15)
    policy.setdefault("lease_ttl_seconds", 120)
    policy.setdefault("tick_timeout_seconds", 0)
    policy.setdefault("reclaim_stale_running_cards", True)
    policy.setdefault("stop_on_human_review", True)
    policy.setdefault("stop_on_task_acceptance", True)
    return policy


def session_id_from_dir(session_dir: str | Path) -> str:
    return Path(session_dir).resolve().name


def select_experiment_card(scan_dir: str | Path) -> Path | None:
    root = Path(scan_dir)
    if not root.exists():
        return None
    for candidate in sorted(root.glob("*.json")):
        card = load_experiment_card(candidate)
        if card.status in {"queued", "stale_requeued"}:
            return candidate
    return None


def validate_phase_gate(card: ExperimentCard, policy: Dict[str, Any]) -> Tuple[bool, str]:
    allowed = policy.get("allowed_phases", [])
    if not allowed:
        return True, ""
    if card.phase in allowed:
        return True, ""
    return False, f"card phase '{card.phase}' is not allowed by controller policy"


def validate_debate_gate(
    card: ExperimentCard, repo_root: str | Path, policy: Dict[str, Any]
) -> Tuple[bool, str]:
    if not card.requires_debate:
        return True, ""
    if not card.debate_bundle_path:
        return False, "debate is required but debate_bundle_path is missing"
    debate_path = resolve_repo_path(repo_root, card.debate_bundle_path)
    if not debate_path.exists():
        return False, f"debate bundle does not exist: {debate_path}"
    debate = load_debate_bundle(debate_path)
    min_rounds = int(policy.get("min_debate_rounds", 2))
    require_artifacts = bool(policy.get("require_debate_artifacts", False))
    return validate_debate_bundle(
        debate,
        repo_root=repo_root,
        min_rounds=min_rounds,
        require_artifacts=require_artifacts,
    )


def should_pause_for_human_review(
    card: ExperimentCard,
    judge_report: Dict[str, Any],
    approval_payload: Dict[str, Any],
    policy: Dict[str, Any],
) -> bool:
    decision = str(judge_report.get("decision", "")).lower()
    if decision != "promote":
        return False
    if not (
        card.human_review_required
        or (
            card.phase_completion_candidate
            and bool(policy.get("human_review_required_for_phase_completion", True))
        )
    ):
        return False
    approved_cards = set(approval_payload.get("approved_cards", []))
    approved_phases = set(approval_payload.get("approved_phases", []))
    if card.experiment_id in approved_cards:
        return False
    if card.phase_completion_candidate and card.phase in approved_phases:
        return False
    return True


def should_pause_for_task_acceptance(
    card: ExperimentCard,
    task_plan: Dict[str, Any] | None,
    approval_payload: Dict[str, Any],
    policy: Dict[str, Any],
) -> bool:
    if not bool(policy.get("stop_on_task_acceptance", True)):
        return False
    if not task_plan:
        return False
    acceptance_status = str(task_plan.get("acceptance_status", "")).lower()
    if acceptance_status not in {"awaiting_human_review", "pending_review"}:
        return False
    approved_cards = set(approval_payload.get("approved_cards", []))
    approved_phases = set(approval_payload.get("approved_phases", []))
    if card.experiment_id in approved_cards:
        return False
    if card.phase in approved_phases:
        return False
    return True


def _append_status_history(
    card: ExperimentCard,
    *,
    from_status: str,
    to_status: str,
    reason: str,
    session_id: str,
) -> None:
    card.status_history.append(
        {
            "from": from_status,
            "to": to_status,
            "reason": reason,
            "session_id": session_id,
            "at_utc": utc_now_iso(),
        }
    )


def transition_card_status(
    card_path: str | Path,
    card: ExperimentCard,
    *,
    new_status: str,
    reason: str,
    session_id: str = "",
) -> ExperimentCard:
    old_status = str(card.status)
    allowed = STATUS_TRANSITIONS.get(old_status, set())
    if old_status != new_status and new_status not in allowed:
        raise ValueError(f"invalid status transition {old_status!r} -> {new_status!r}")
    if old_status != new_status:
        _append_status_history(
            card,
            from_status=old_status,
            to_status=new_status,
            reason=reason,
            session_id=session_id,
        )
    card.status = new_status
    write_experiment_card(card_path, card)
    return card


def lease_dir(repo_root: str | Path, policy: Dict[str, Any]) -> Path:
    path = resolve_repo_path(repo_root, str(policy.get("lease_dir")))
    path.mkdir(parents=True, exist_ok=True)
    return path


def lease_path(repo_root: str | Path, policy: Dict[str, Any], experiment_id: str) -> Path:
    return lease_dir(repo_root, policy) / f"{experiment_id}.json"


def load_lease(repo_root: str | Path, policy: Dict[str, Any], experiment_id: str) -> Dict[str, Any] | None:
    path = lease_path(repo_root, policy, experiment_id)
    if not path.exists():
        return None
    return load_json(path)


def is_lease_stale(lease_payload: Dict[str, Any] | None, policy: Dict[str, Any]) -> bool:
    if not lease_payload:
        return True
    last_heartbeat = str(lease_payload.get("last_heartbeat_at_utc", ""))
    if not last_heartbeat:
        return True
    return is_utc_iso_stale(
        last_heartbeat,
        ttl_seconds=float(policy.get("lease_ttl_seconds", 120)),
    )


def write_lease(
    repo_root: str | Path,
    policy: Dict[str, Any],
    *,
    experiment_id: str,
    session_id: str,
    status: str,
    current_step: str,
) -> Path:
    now = utc_now_iso()
    ttl_seconds = float(policy.get("lease_ttl_seconds", 120))
    payload = {
        "experiment_id": experiment_id,
        "session_id": session_id,
        "status": status,
        "current_step": current_step,
        "claimed_at_utc": now,
        "last_heartbeat_at_utc": now,
        "lease_expires_at_utc": add_seconds_iso(now, ttl_seconds),
    }
    target = lease_path(repo_root, policy, experiment_id)
    write_json(target, payload)
    return target


def acquire_experiment_lease(
    repo_root: str | Path,
    policy: Dict[str, Any],
    *,
    experiment_id: str,
    session_id: str,
    current_step: str,
    runtime_profile_id: str = "",
) -> tuple[bool, str]:
    now = utc_now_iso()
    path = lease_path(repo_root, policy, experiment_id)
    payload = {
        "experiment_id": experiment_id,
        "session_id": session_id,
        "status": "claimed",
        "current_step": current_step,
        "runtime_profile_id": runtime_profile_id,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "process_cmdline": " ".join(sys.argv),
        "claimed_at_utc": now,
        "process_started_at_utc": now,
    }
    payload["last_heartbeat_at_utc"] = payload["claimed_at_utc"]
    payload["lease_expires_at_utc"] = add_seconds_iso(
        payload["claimed_at_utc"],
        float(policy.get("lease_ttl_seconds", 120)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            import json

            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return True, "claimed new lease"
    except FileExistsError:
        existing = load_json(path)
        if is_lease_stale(existing, policy):
            write_json(path, payload)
            return True, "reclaimed stale lease"
        return False, f"active lease owned by session {existing.get('session_id')}"


def refresh_experiment_lease(
    repo_root: str | Path,
    policy: Dict[str, Any],
    *,
    experiment_id: str,
    session_id: str,
    status: str,
    current_step: str,
    runtime_profile_id: str = "",
) -> Path:
    now = utc_now_iso()
    payload = load_lease(repo_root, policy, experiment_id) or {}
    payload.update(
        {
            "experiment_id": experiment_id,
            "session_id": session_id,
            "status": status,
            "current_step": current_step,
            "runtime_profile_id": runtime_profile_id or payload.get("runtime_profile_id", ""),
            "hostname": payload.get("hostname", socket.gethostname()),
            "pid": int(payload.get("pid", os.getpid())),
            "process_cmdline": payload.get("process_cmdline", " ".join(sys.argv)),
            "last_heartbeat_at_utc": now,
            "lease_expires_at_utc": add_seconds_iso(
                now,
                float(policy.get("lease_ttl_seconds", 120)),
            ),
        }
    )
    target = lease_path(repo_root, policy, experiment_id)
    write_json(target, payload)
    return target


def release_experiment_lease(repo_root: str | Path, policy: Dict[str, Any], experiment_id: str) -> None:
    path = lease_path(repo_root, policy, experiment_id)
    if path.exists():
        path.unlink()


def _local_process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _local_process_cmdline(pid: int) -> str:
    proc_path = Path(f"/proc/{int(pid)}/cmdline")
    if not proc_path.exists():
        return ""
    try:
        raw = proc_path.read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def lease_process_is_alive(lease_payload: Dict[str, Any] | None) -> bool:
    if not lease_payload:
        return False
    hostname = str(lease_payload.get("hostname", ""))
    if hostname and hostname != socket.gethostname():
        return False
    try:
        pid = int(lease_payload.get("pid", 0))
    except (TypeError, ValueError):
        return False
    if not _local_process_is_alive(pid):
        return False
    expected_cmdline = str(lease_payload.get("process_cmdline", "")).strip()
    if not expected_cmdline:
        return True
    actual_cmdline = _local_process_cmdline(pid)
    if not actual_cmdline:
        return False
    if "run_research_queue.py" in expected_cmdline:
        return "run_research_queue.py" in actual_cmdline
    return True


def reclaim_stale_experiment_cards(scan_dir: str | Path, repo_root: str | Path, policy: Dict[str, Any]) -> list[str]:
    if not bool(policy.get("reclaim_stale_running_cards", True)):
        return []
    reclaimed: list[str] = []
    root = Path(scan_dir)
    if not root.exists():
        return reclaimed
    for candidate in sorted(root.glob("*.json")):
        card = load_experiment_card(candidate)
        if card.status not in RUNNING_LIKE_STATUSES:
            continue
        lease_payload = load_lease(repo_root, policy, card.experiment_id)
        if not is_lease_stale(lease_payload, policy):
            continue
        if lease_process_is_alive(lease_payload):
            continue
        transition_card_status(
            candidate,
            card,
            new_status="stale_requeued",
            reason="controller reclaimed stale card lease",
            session_id="controller-reclaim",
        )
        release_experiment_lease(repo_root, policy, card.experiment_id)
        reclaimed.append(card.experiment_id)
    return reclaimed


def release_human_approved_cards(
    scan_dir: str | Path,
    approval_payload: Dict[str, Any],
) -> list[str]:
    released: list[str] = []
    approved_cards = {str(item) for item in approval_payload.get("approved_cards", [])}
    approved_phases = {str(item) for item in approval_payload.get("approved_phases", [])}
    requeue_cards = {str(item) for item in approval_payload.get("requeue_cards", [])}
    requeue_phases = {str(item) for item in approval_payload.get("requeue_phases", [])}

    root = Path(scan_dir)
    if not root.exists():
        return released
    for candidate in sorted(root.glob("*.json")):
        card = load_experiment_card(candidate)
        if card.status != "awaiting_human_review":
            continue
        if card.experiment_id in requeue_cards or card.phase in requeue_phases:
            transition_card_status(
                candidate,
                card,
                new_status="queued",
                reason="human review requested requeue",
                session_id="human-review-release",
            )
            released.append(card.experiment_id)
            continue
        if card.experiment_id in approved_cards or card.phase in approved_phases:
            transition_card_status(
                candidate,
                card,
                new_status="completed",
                reason="human review approved result",
                session_id="human-review-release",
            )
            released.append(card.experiment_id)
    return released


def write_session_manifest(
    session_dir: str | Path,
    *,
    session_id: str,
    controller_policy_path: str,
    memory_paths: list[str],
    selected_card_path: str,
    human_approval_path: str,
) -> Path:
    payload = {
        "session_id": session_id,
        "controller_policy_path": controller_policy_path,
        "memory_paths": memory_paths,
        "selected_card_path": selected_card_path,
        "human_approval_path": human_approval_path,
        "started_at_utc": utc_now_iso(),
    }
    target = Path(session_dir) / "session_manifest.json"
    write_json(target, payload)
    return target


def write_session_checkpoint(
    session_dir: str | Path,
    *,
    status: str,
    experiment_id: str,
    card_path: str,
    reason: str,
    controller_policy_path: str,
    memory_paths: list[str],
    session_id: str = "",
    current_step: str = "",
    last_heartbeat_at_utc: str = "",
    lease_expires_at_utc: str = "",
    recovery_hint: str = "",
) -> Path:
    checkpoint = {
        "status": status,
        "experiment_id": experiment_id,
        "card_path": card_path,
        "reason": reason,
        "controller_policy_path": controller_policy_path,
        "memory_paths": memory_paths,
        "session_id": session_id,
        "current_step": current_step,
        "last_heartbeat_at_utc": last_heartbeat_at_utc,
        "lease_expires_at_utc": lease_expires_at_utc,
        "recovery_hint": recovery_hint,
        "updated_at_utc": utc_now_iso(),
    }
    target = Path(session_dir) / "session_checkpoint.json"
    write_json(target, checkpoint)
    return target


def write_session_heartbeat(
    session_dir: str | Path,
    *,
    session_id: str,
    experiment_id: str,
    status: str,
    current_step: str,
    lease_expires_at_utc: str,
) -> Path:
    payload = {
        "session_id": session_id,
        "experiment_id": experiment_id,
        "status": status,
        "current_step": current_step,
        "last_heartbeat_at_utc": utc_now_iso(),
        "lease_expires_at_utc": lease_expires_at_utc,
    }
    target = Path(session_dir) / "heartbeat.json"
    write_json(target, payload)
    return target


def write_session_error(
    session_dir: str | Path,
    *,
    experiment_id: str,
    session_id: str,
    error_kind: str,
    message: str,
    traceback_text: str,
) -> Path:
    payload = {
        "experiment_id": experiment_id,
        "session_id": session_id,
        "error_kind": error_kind,
        "message": message,
        "traceback": traceback_text,
        "recorded_at_utc": utc_now_iso(),
    }
    target = Path(session_dir) / "session_error.json"
    write_json(target, payload)
    return target


def new_session_dir(repo_root: str | Path, policy: Dict[str, Any]) -> Path:
    session_root = resolve_repo_path(repo_root, str(policy.get("session_output_dir")))
    session_name = f"session_{utc_now_iso().replace(':', '-').replace('+00:00', 'Z')}_{os.getpid()}"
    session_dir = session_root / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir
