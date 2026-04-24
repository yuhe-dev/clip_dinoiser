"""Queue runner with code-enforced debate, phase, lease, and human-review gates."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    ROOT = os.path.abspath(os.path.dirname(__file__))
    PARENT = os.path.dirname(ROOT)
    if PARENT not in sys.path:
        sys.path.insert(0, PARENT)

from clip_dinoiser.research_harness.controller import (
    acquire_experiment_lease,
    discover_repo_root,
    load_controller_policy,
    load_lease,
    new_session_dir,
    reclaim_stale_experiment_cards,
    refresh_experiment_lease,
    release_experiment_lease,
    select_experiment_card,
    session_id_from_dir,
    should_pause_for_human_review,
    should_pause_for_task_acceptance,
    transition_card_status,
    validate_debate_gate,
    validate_phase_gate,
    write_session_checkpoint,
    write_session_error,
    write_session_heartbeat,
    write_session_manifest,
)
from clip_dinoiser.research_harness.attempts import allocate_attempt_id, finalize_attempt, start_attempt
from clip_dinoiser.research_harness.absorber import build_runtime_index, write_runtime_index
from clip_dinoiser.research_harness.context_packet import write_context_packet
from clip_dinoiser.research_harness.loop_catalog import card_is_execution_ready, execution_readiness_reason
from clip_dinoiser.research_harness.preflight import card_requires_worker_runtime, resolve_runtime_selection, write_preflight_report
from clip_dinoiser.research_harness.registry import load_experiment_card, load_json, resolve_repo_path, write_experiment_card
from clip_dinoiser.research_harness.scheduler import select_experiment_card
from clip_dinoiser.run_research_tick import main as run_research_tick_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run controller-enforced research queue steps.")
    parser.add_argument("--experiment-card", help="Optional explicit experiment card path.")
    parser.add_argument(
        "--scan-dir",
        default=".slicetune/experiments",
        help="Directory to scan for queued experiment cards when --experiment-card is omitted.",
    )
    parser.add_argument(
        "--controller-policy",
        default=".slicetune/runtime/controller_policy.json",
        help="Machine-readable controller policy path.",
    )
    parser.add_argument(
        "--human-approval",
        default=".slicetune/approvals/human_review.json",
        help="Machine-readable human approval file.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1,
        help="Maximum number of cards to process in this controller invocation. Use 0 for unbounded.",
    )
    parser.add_argument(
        "--until-blocked",
        action="store_true",
        help="Keep processing queued cards until blocked, human review, or queue exhaustion.",
    )
    return parser


def _select_card(args: argparse.Namespace, repo_root: Path) -> Path:
    if args.experiment_card:
        return Path(args.experiment_card)
    selected = select_experiment_card(resolve_repo_path(repo_root, args.scan_dir))
    if selected is None:
        raise FileNotFoundError("no queued experiment card found")
    return selected


def _heartbeat_loop(
    *,
    stop_event: threading.Event,
    repo_root: Path,
    policy: dict,
    session_dir: Path,
    session_id: str,
    experiment_id: str,
    controller_policy_path: str,
    card_path: str,
    memory_paths: list[str],
    interval_seconds: float,
    runtime_profile_id: str,
) -> None:
    while not stop_event.wait(interval_seconds):
        refresh_experiment_lease(
            repo_root,
            policy,
            experiment_id=experiment_id,
            session_id=session_id,
            status="running",
            current_step="run_research_tick",
            runtime_profile_id=runtime_profile_id,
        )
        lease_payload = load_lease(repo_root, policy, experiment_id) or {}
        write_session_checkpoint(
            session_dir,
            status="running",
            experiment_id=experiment_id,
            card_path=card_path,
            reason="heartbeat refresh",
            controller_policy_path=controller_policy_path,
            memory_paths=memory_paths,
            session_id=session_id,
            current_step="run_research_tick",
            last_heartbeat_at_utc=str(lease_payload.get("last_heartbeat_at_utc", "")),
            lease_expires_at_utc=str(lease_payload.get("lease_expires_at_utc", "")),
            recovery_hint="safe to rerun card if lease becomes stale; completed outputs are idempotent",
        )
        write_session_heartbeat(
            session_dir,
            session_id=session_id,
            experiment_id=experiment_id,
            status="running",
            current_step="run_research_tick",
            lease_expires_at_utc=str(lease_payload.get("lease_expires_at_utc", "")),
        )


def _run_tick_with_watchdog(
    *,
    repo_root: Path,
    card_path: Path,
    session_dir: Path,
    runtime_profile_id: str,
    resolved_python_bin: str,
    timeout_seconds: float,
) -> tuple[int, Exception | None, str, dict]:
    if timeout_seconds <= 0:
        tick_args = ["--experiment-card", str(card_path)]
        if resolved_python_bin:
            tick_args.extend(["--python-bin-override", resolved_python_bin])
        if runtime_profile_id:
            tick_args.extend(["--runtime-profile-id", runtime_profile_id])
        return run_research_tick_main(tick_args), None, "", {}

    tick_stdout_path = session_dir / "tick.stdout.log"
    tick_stderr_path = session_dir / "tick.stderr.log"
    command = [
        sys.executable,
        str((repo_root / "run_research_tick.py").resolve()),
        "--experiment-card",
        str(card_path.resolve()),
    ]
    if resolved_python_bin:
        command.extend(["--python-bin-override", resolved_python_bin])
    if runtime_profile_id:
        command.extend(["--runtime-profile-id", runtime_profile_id])

    with tick_stdout_path.open("a", encoding="utf-8") as stdout_handle, tick_stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            stdout=stdout_handle,
            stderr=stderr_handle,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        try:
            exit_code = process.wait(timeout=timeout_seconds)
            return exit_code, None, "", {
                "tick_stdout_path": str(tick_stdout_path.resolve()),
                "tick_stderr_path": str(tick_stderr_path.resolve()),
            }
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            message = f"tick subprocess exceeded timeout of {timeout_seconds:.1f}s"
            return 124, TimeoutError(message), "", {
                "tick_stdout_path": str(tick_stdout_path.resolve()),
                "tick_stderr_path": str(tick_stderr_path.resolve()),
            }


def _run_single_card(
    *,
    card_path: Path,
    repo_root: Path,
    policy_path: Path,
    policy: dict,
    human_approval_path: Path,
    memory_paths: list[str],
) -> int:
    card = load_experiment_card(card_path)
    output_dir = resolve_repo_path(repo_root, card.output_dir)
    session_dir = new_session_dir(repo_root, policy)
    session_id = session_id_from_dir(session_dir)
    runtime_index_path = resolve_repo_path(repo_root, ".slicetune/state/runtime_index.json")

    write_session_manifest(
        session_dir,
        session_id=session_id,
        controller_policy_path=str(policy_path.resolve()),
        memory_paths=memory_paths,
        selected_card_path=str(card_path.resolve()),
        human_approval_path=str(human_approval_path.resolve()),
    )
    write_context_packet(
        session_dir,
        repo_root=repo_root,
        card=card,
        card_path=card_path,
        memory_paths=memory_paths,
    )

    if card.status not in {"queued", "stale_requeued"}:
        write_session_checkpoint(
            session_dir,
            status="skipped_non_queueable_status",
            experiment_id=card.experiment_id,
            card_path=str(card_path.resolve()),
            reason=f"card status '{card.status}' is not queueable",
            controller_policy_path=str(policy_path.resolve()),
            memory_paths=memory_paths,
            session_id=session_id,
            current_step="preflight",
            recovery_hint="manually reset the card to queued if you intend to rerun it",
        )
        print(f"{card.experiment_id}: skipped_non_queueable_status: {card.status}")
        return 6

    if int(card.max_attempts) > 0 and int(card.attempt_count) >= int(card.max_attempts):
        transition_card_status(
            card_path,
            card,
            new_status="blocked_retry_limit",
            reason=f"attempt_count {card.attempt_count} reached max_attempts {card.max_attempts}",
            session_id=session_id,
        )
        write_session_checkpoint(
            session_dir,
            status="blocked_retry_limit",
            experiment_id=card.experiment_id,
            card_path=str(card_path.resolve()),
            reason=f"attempt_count {card.attempt_count} reached max_attempts {card.max_attempts}",
            controller_policy_path=str(policy_path.resolve()),
            memory_paths=memory_paths,
            session_id=session_id,
            current_step="retry_guard",
            recovery_hint="raise max_attempts or inspect prior attempts before requeueing",
        )
        print(f"{card.experiment_id}: blocked_retry_limit")
        return 9

    ok, reason = validate_phase_gate(card, policy)
    if not ok:
        transition_card_status(
            card_path,
            card,
            new_status="blocked_phase_gate",
            reason=reason,
            session_id=session_id,
        )
        write_session_checkpoint(
            session_dir,
            status="blocked_phase_gate",
            experiment_id=card.experiment_id,
            card_path=str(card_path.resolve()),
            reason=reason,
            controller_policy_path=str(policy_path.resolve()),
            memory_paths=memory_paths,
            session_id=session_id,
            current_step="phase_gate",
            recovery_hint="update phase or controller policy before retry",
        )
        print(f"{card.experiment_id}: blocked_phase_gate: {reason}")
        write_runtime_index(runtime_index_path, build_runtime_index(repo_root, resolve_repo_path(repo_root, ".slicetune/experiments")))
        return 3

    ok, reason = validate_debate_gate(card, repo_root, policy)
    if not ok:
        transition_card_status(
            card_path,
            card,
            new_status="blocked_debate_gate",
            reason=reason,
            session_id=session_id,
        )
        write_session_checkpoint(
            session_dir,
            status="blocked_debate_gate",
            experiment_id=card.experiment_id,
            card_path=str(card_path.resolve()),
            reason=reason,
            controller_policy_path=str(policy_path.resolve()),
            memory_paths=memory_paths,
            session_id=session_id,
            current_step="debate_gate",
            recovery_hint="write an approved debate bundle before retry",
        )
        print(f"{card.experiment_id}: blocked_debate_gate: {reason}")
        write_runtime_index(runtime_index_path, build_runtime_index(repo_root, resolve_repo_path(repo_root, ".slicetune/experiments")))
        return 4

    if not card_is_execution_ready(card):
        reason = execution_readiness_reason(card)
        transition_card_status(
            card_path,
            card,
            new_status="blocked_preflight",
            reason=reason,
            session_id=session_id,
        )
        write_session_checkpoint(
            session_dir,
            status="blocked_preflight",
            experiment_id=card.experiment_id,
            card_path=str(card_path.resolve()),
            reason=reason,
            controller_policy_path=str(policy_path.resolve()),
            memory_paths=memory_paths,
            session_id=session_id,
            current_step="execution_readiness",
            recovery_hint="implement an execution handler or mark a different executable loop before requeueing",
        )
        print(f"{card.experiment_id}: blocked_preflight: {reason}")
        write_runtime_index(runtime_index_path, build_runtime_index(repo_root, resolve_repo_path(repo_root, ".slicetune/experiments")))
        return 7

    runtime_profile_id = ""
    resolved_python_bin = ""
    attempt_id = ""
    attempt_dir = None
    if card_requires_worker_runtime(card):
        preflight_report = resolve_runtime_selection(
            card,
            repo_root=repo_root,
            policy=policy,
        )
        write_preflight_report(session_dir, preflight_report)
        if preflight_report.get("status") != "passed":
            transition_card_status(
                card_path,
                card,
                new_status="blocked_preflight",
                reason=str(preflight_report.get("reason", "worker runtime preflight failed")),
                session_id=session_id,
            )
            write_session_checkpoint(
                session_dir,
                status="blocked_preflight",
                experiment_id=card.experiment_id,
                card_path=str(card_path.resolve()),
                reason=str(preflight_report.get("reason", "worker runtime preflight failed")),
                controller_policy_path=str(policy_path.resolve()),
                memory_paths=memory_paths,
                session_id=session_id,
                current_step="preflight",
                recovery_hint="fix runtime profiles / dependencies and requeue card",
            )
            print(f"{card.experiment_id}: blocked_preflight: {preflight_report.get('reason', 'unknown reason')}")
            write_runtime_index(runtime_index_path, build_runtime_index(repo_root, resolve_repo_path(repo_root, ".slicetune/experiments")))
            return 7
        runtime_profile_id = str(preflight_report.get("selected_profile_id", ""))
        resolved_python_bin = str(preflight_report.get("selected_python_bin", ""))

    claimed, claim_reason = acquire_experiment_lease(
        repo_root,
        policy,
        experiment_id=card.experiment_id,
        session_id=session_id,
        current_step="claim",
        runtime_profile_id=runtime_profile_id,
    )
    if not claimed:
        print(f"{card.experiment_id}: lease_busy: {claim_reason}")
        return 5

    try:
        attempt_id = allocate_attempt_id(card.experiment_id)
        card.attempt_count += 1
        card.last_attempt_id = attempt_id
        write_experiment_card(card_path, card)
        attempt_dir = start_attempt(
            attempt_id=attempt_id,
            card=card,
            card_path=card_path,
            session_id=session_id,
            output_dir=output_dir,
            runtime_profile_id=runtime_profile_id,
            python_bin=resolved_python_bin,
            metadata={
                "controller_policy_path": str(policy_path.resolve()),
                "session_dir": str(session_dir.resolve()),
            },
        )
        transition_card_status(
            card_path,
            card,
            new_status="claimed",
            reason=claim_reason,
            session_id=session_id,
        )
        transition_card_status(
            card_path,
            card,
            new_status="running",
            reason="controller gates passed",
            session_id=session_id,
        )
        lease_payload = load_lease(repo_root, policy, card.experiment_id) or {}
        write_session_checkpoint(
            session_dir,
            status="running",
            experiment_id=card.experiment_id,
            card_path=str(card_path.resolve()),
            reason="controller gates passed",
            controller_policy_path=str(policy_path.resolve()),
            memory_paths=memory_paths,
            session_id=session_id,
            current_step="run_research_tick",
            last_heartbeat_at_utc=str(lease_payload.get("last_heartbeat_at_utc", "")),
            lease_expires_at_utc=str(lease_payload.get("lease_expires_at_utc", "")),
            recovery_hint="safe to rerun card if lease becomes stale; completed outputs are idempotent",
        )
        write_session_heartbeat(
            session_dir,
            session_id=session_id,
            experiment_id=card.experiment_id,
            status="running",
            current_step="run_research_tick",
            lease_expires_at_utc=str(lease_payload.get("lease_expires_at_utc", "")),
        )

        stop_event = threading.Event()
        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            kwargs={
                "stop_event": stop_event,
                "repo_root": repo_root,
                "policy": policy,
                "session_dir": session_dir,
                "session_id": session_id,
                "experiment_id": card.experiment_id,
                "controller_policy_path": str(policy_path.resolve()),
                "card_path": str(card_path.resolve()),
                "memory_paths": memory_paths,
                "interval_seconds": float(policy.get("heartbeat_interval_seconds", 15)),
                "runtime_profile_id": runtime_profile_id,
            },
            daemon=True,
        )
        heartbeat_thread.start()
        tick_exception: Exception | None = None
        tick_traceback = ""
        tick_artifacts: dict = {}
        try:
            exit_code, tick_exception, tick_traceback, tick_artifacts = _run_tick_with_watchdog(
                repo_root=repo_root,
                card_path=card_path,
                session_dir=session_dir,
                runtime_profile_id=runtime_profile_id,
                resolved_python_bin=resolved_python_bin,
                timeout_seconds=float(policy.get("tick_timeout_seconds", 0)),
            )
        except Exception as exc:  # pragma: no cover - exercised by CLI tests via patching
            tick_exception = exc
            tick_traceback = traceback.format_exc()
            exit_code = 8
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=1.0)

        card = load_experiment_card(card_path)
        if exit_code != 0:
            failure_reason = (
                f"tick raised {type(tick_exception).__name__}: {tick_exception}"
                if tick_exception is not None
                else f"tick exited with code {exit_code}"
            )
            transition_card_status(
                card_path,
                card,
                new_status="failed_execution",
                reason=failure_reason,
                session_id=session_id,
            )
            lease_payload = load_lease(repo_root, policy, card.experiment_id) or {}
            if tick_exception is not None:
                write_session_error(
                    session_dir,
                    experiment_id=card.experiment_id,
                    session_id=session_id,
                    error_kind=type(tick_exception).__name__,
                    message=str(tick_exception),
                    traceback_text=tick_traceback,
                )
            write_session_checkpoint(
                session_dir,
                status="failed_execution",
                experiment_id=card.experiment_id,
                card_path=str(card_path.resolve()),
                reason=failure_reason,
                controller_policy_path=str(policy_path.resolve()),
                memory_paths=memory_paths,
                session_id=session_id,
                current_step="run_research_tick",
                last_heartbeat_at_utc=str(lease_payload.get("last_heartbeat_at_utc", "")),
                lease_expires_at_utc=str(lease_payload.get("lease_expires_at_utc", "")),
                recovery_hint="fix failure and requeue card",
            )
            if attempt_dir is not None:
                finalize_attempt(
                    attempt_dir=attempt_dir,
                    attempt_id=attempt_id,
                    card=card,
                    card_path=card_path,
                    session_id=session_id,
                    output_dir=output_dir,
                    runtime_profile_id=runtime_profile_id,
                    python_bin=resolved_python_bin,
                    status="failed_execution",
                    reason=failure_reason,
                    exit_code=exit_code,
                    paths={
                        "session_dir": str(session_dir.resolve()),
                        "session_checkpoint_path": str((session_dir / "session_checkpoint.json").resolve()),
                        "session_error_path": str((session_dir / "session_error.json").resolve()),
                        "preflight_report_path": str((session_dir / "preflight_report.json").resolve()),
                        **tick_artifacts,
                    },
                )
            write_runtime_index(runtime_index_path, build_runtime_index(repo_root, resolve_repo_path(repo_root, ".slicetune/experiments")))
            return exit_code

        judge_report = load_json(output_dir / "judge_report.json")
        task_plan_payload = {}
        task_plan_path = output_dir / "task_plan.json"
        if task_plan_path.exists():
            task_plan_payload = load_json(task_plan_path)
        approvals = (
            load_json(human_approval_path)
            if human_approval_path.exists()
            else {"approved_cards": [], "approved_phases": []}
        )
        if should_pause_for_human_review(card, judge_report, approvals, policy) or should_pause_for_task_acceptance(
            card,
            task_plan_payload,
            approvals,
            policy,
        ):
            transition_card_status(
                card_path,
                card,
                new_status="awaiting_human_review",
                reason="controller stop after successful run; waiting for human acceptance",
                session_id=session_id,
            )
            lease_payload = load_lease(repo_root, policy, card.experiment_id) or {}
            write_session_checkpoint(
                session_dir,
                status="awaiting_human_review",
                experiment_id=card.experiment_id,
                card_path=str(card_path.resolve()),
                reason="controller stop after successful run; waiting for human acceptance",
                controller_policy_path=str(policy_path.resolve()),
                memory_paths=memory_paths,
                session_id=session_id,
                current_step="human_review_gate",
                last_heartbeat_at_utc=str(lease_payload.get("last_heartbeat_at_utc", "")),
                lease_expires_at_utc=str(lease_payload.get("lease_expires_at_utc", "")),
                recovery_hint="resume only after human approval file is updated",
            )
            if attempt_dir is not None:
                finalize_attempt(
                    attempt_dir=attempt_dir,
                    attempt_id=attempt_id,
                    card=card,
                    card_path=card_path,
                    session_id=session_id,
                    output_dir=output_dir,
                    runtime_profile_id=runtime_profile_id,
                    python_bin=resolved_python_bin,
                    status="awaiting_human_review",
                    reason="waiting for human acceptance",
                    exit_code=10,
                    paths={
                        "session_dir": str(session_dir.resolve()),
                        "session_checkpoint_path": str((session_dir / "session_checkpoint.json").resolve()),
                        "preflight_report_path": str((session_dir / "preflight_report.json").resolve()),
                        "judge_report_path": str((output_dir / "judge_report.json").resolve()),
                        "result_bundle_path": str((output_dir / "result_bundle.json").resolve()),
                        "run_manifest_path": str((output_dir / "run_manifest.json").resolve()),
                        **tick_artifacts,
                    },
                )
            print(f"{card.experiment_id}: awaiting_human_review")
            write_runtime_index(runtime_index_path, build_runtime_index(repo_root, resolve_repo_path(repo_root, ".slicetune/experiments")))
            return 10

        transition_card_status(
            card_path,
            card,
            new_status="completed",
            reason="controller completed run without additional stop gates",
            session_id=session_id,
        )
        lease_payload = load_lease(repo_root, policy, card.experiment_id) or {}
        write_session_checkpoint(
            session_dir,
            status="completed",
            experiment_id=card.experiment_id,
            card_path=str(card_path.resolve()),
            reason="controller completed run without additional stop gates",
            controller_policy_path=str(policy_path.resolve()),
            memory_paths=memory_paths,
            session_id=session_id,
            current_step="done",
            last_heartbeat_at_utc=str(lease_payload.get("last_heartbeat_at_utc", "")),
            lease_expires_at_utc=str(lease_payload.get("lease_expires_at_utc", "")),
            recovery_hint="none",
        )
        if attempt_dir is not None:
            finalize_attempt(
                attempt_dir=attempt_dir,
                attempt_id=attempt_id,
                card=card,
                card_path=card_path,
                session_id=session_id,
                output_dir=output_dir,
                runtime_profile_id=runtime_profile_id,
                python_bin=resolved_python_bin,
                status="completed",
                reason="controller completed run without additional stop gates",
                exit_code=0,
                paths={
                    "session_dir": str(session_dir.resolve()),
                    "session_checkpoint_path": str((session_dir / "session_checkpoint.json").resolve()),
                    "preflight_report_path": str((session_dir / "preflight_report.json").resolve()),
                    "judge_report_path": str((output_dir / "judge_report.json").resolve()),
                    "result_bundle_path": str((output_dir / "result_bundle.json").resolve()),
                    "run_manifest_path": str((output_dir / "run_manifest.json").resolve()),
                    **tick_artifacts,
                },
            )
        print(f"{card.experiment_id}: completed")
        write_runtime_index(runtime_index_path, build_runtime_index(repo_root, resolve_repo_path(repo_root, ".slicetune/experiments")))
        return 0
    finally:
        release_experiment_lease(repo_root, policy, card.experiment_id)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    seed_path = Path(args.experiment_card) if args.experiment_card else Path(args.scan_dir)
    repo_root = discover_repo_root(seed_path)
    policy_path = resolve_repo_path(repo_root, args.controller_policy)
    policy = load_controller_policy(policy_path)
    human_approval_path = resolve_repo_path(repo_root, args.human_approval)
    scan_dir = resolve_repo_path(repo_root, args.scan_dir)
    memory_paths = [str(resolve_repo_path(repo_root, path)) for path in policy.get("always_load_paths", [])]

    steps = 0
    max_steps = int(args.max_steps)
    while True:
        reclaim_stale_experiment_cards(scan_dir, repo_root, policy)
        try:
            card_path = _select_card(args, repo_root)
        except FileNotFoundError:
            print("queue_empty")
            return 0

        exit_code = _run_single_card(
            card_path=card_path,
            repo_root=repo_root,
            policy_path=policy_path,
            policy=policy,
            human_approval_path=human_approval_path,
            memory_paths=memory_paths,
        )
        steps += 1

        if exit_code == 10 and bool(policy.get("stop_on_human_review", True)):
            return exit_code
        if exit_code not in {0, 10}:
            return exit_code
        if args.experiment_card:
            return exit_code
        if max_steps > 0 and steps >= max_steps:
            return exit_code
        if not args.until_blocked:
            return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
