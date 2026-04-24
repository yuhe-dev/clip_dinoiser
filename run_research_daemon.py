"""Long-running daemon loop for the autonomous research harness."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    ROOT = os.path.abspath(os.path.dirname(__file__))
    PARENT = os.path.dirname(ROOT)
    if PARENT not in sys.path:
        sys.path.insert(0, PARENT)

from clip_dinoiser.research_harness.controller import (
    discover_repo_root,
    load_controller_policy,
    release_human_approved_cards,
    reclaim_stale_experiment_cards,
)
from clip_dinoiser.research_harness.agentic import ensure_agentic_artifacts
from clip_dinoiser.research_harness.absorber import build_runtime_index, write_runtime_index
from clip_dinoiser.research_harness.debate import auto_generate_debate_bundle
from clip_dinoiser.research_harness.registry import load_experiment_card, load_json, resolve_repo_path, write_json
from clip_dinoiser.research_harness.runtime import utc_now_iso
from clip_dinoiser.research_harness.scheduler import build_queue_snapshot, select_experiment_card, write_queue_snapshot
from clip_dinoiser.research_harness.task_board import build_task_board
from clip_dinoiser.research_harness.taskflow import auto_advance_card, reconcile_task_plans
from clip_dinoiser.run_research_propose import main as run_research_propose_main
from clip_dinoiser.run_research_queue import main as run_research_queue_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the research harness queue as a long-lived daemon loop.")
    parser.add_argument("--scan-dir", default=".slicetune/experiments")
    parser.add_argument("--controller-policy", default=".slicetune/runtime/controller_policy.json")
    parser.add_argument("--human-approval", default=".slicetune/approvals/human_review.json")
    parser.add_argument("--poll-interval-seconds", type=float, default=30.0)
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means unbounded")
    parser.add_argument("--max-idle-cycles", type=int, default=0, help="0 means unbounded")
    parser.add_argument("--continue-after-failure", action="store_true")
    parser.add_argument("--status-path", default="artifacts/research_harness/daemon/daemon_status.json")
    parser.add_argument("--queue-snapshot-path", default="artifacts/research_harness/daemon/queue_snapshot.json")
    parser.add_argument("--runtime-index-path", default=".slicetune/state/runtime_index.json")
    parser.add_argument("--task-board-path", default=".slicetune/state/task_board.json")
    parser.add_argument("--taskflow-index-path", default=".slicetune/state/taskflow_index.json")
    parser.add_argument("--proposal-policy", default=".slicetune/runtime/proposal_policy.json")
    parser.add_argument("--proposal-index-path", default=".slicetune/state/proposal_index.json")
    parser.add_argument("--proposals-dir", default=".slicetune/proposals")
    parser.add_argument("--auto-propose", action="store_true")
    parser.add_argument("--auto-debate", action="store_true")
    parser.add_argument("--auto-agentic", action="store_true")
    return parser


def write_daemon_status(path: str | Path, payload: dict) -> Path:
    write_json(path, payload)
    return Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = discover_repo_root(Path(args.scan_dir))
    policy_path = resolve_repo_path(repo_root, args.controller_policy)
    policy = load_controller_policy(policy_path)
    scan_dir = resolve_repo_path(repo_root, args.scan_dir)
    human_approval_path = resolve_repo_path(repo_root, args.human_approval)
    status_path = resolve_repo_path(repo_root, args.status_path)
    queue_snapshot_path = resolve_repo_path(repo_root, args.queue_snapshot_path)
    runtime_index_path = resolve_repo_path(repo_root, args.runtime_index_path)
    task_board_path = resolve_repo_path(repo_root, args.task_board_path)
    taskflow_index_path = resolve_repo_path(repo_root, args.taskflow_index_path)
    proposal_policy_path = resolve_repo_path(repo_root, args.proposal_policy)
    proposal_index_path = resolve_repo_path(repo_root, args.proposal_index_path)
    proposals_dir = resolve_repo_path(repo_root, args.proposals_dir)

    cycles = 0
    idle_cycles = 0
    started_at = utc_now_iso()

    while True:
        cycles += 1
        reclaim_stale_experiment_cards(scan_dir, repo_root, policy)
        if human_approval_path.exists():
            release_human_approved_cards(scan_dir, load_json(human_approval_path))
        if args.auto_agentic:
            for candidate in sorted(scan_dir.glob("*.json")):
                card = load_experiment_card(candidate)
                ensure_agentic_artifacts(
                    repo_root=repo_root,
                    card=card,
                    card_path=candidate,
                    execute_literature_search=False,
                )
        if args.auto_debate:
            for candidate in sorted(scan_dir.glob("*.json")):
                card = load_experiment_card(candidate)
                if not card.requires_debate or not card.debate_bundle_path:
                    continue
                if card.status not in {"planned", "blocked_debate_gate"}:
                    continue
                auto_generate_debate_bundle(card=card, repo_root=repo_root)
        for candidate in sorted(scan_dir.glob("*.json")):
            auto_advance_card(candidate, repo_root=repo_root, controller_policy_path=policy_path)
        write_json(taskflow_index_path, reconcile_task_plans(scan_dir, repo_root=repo_root))
        snapshot = build_queue_snapshot(scan_dir)
        write_queue_snapshot(queue_snapshot_path, snapshot)
        write_runtime_index(runtime_index_path, build_runtime_index(repo_root, scan_dir))
        write_json(task_board_path, build_task_board(repo_root, scan_dir))
        if args.auto_propose:
            run_research_propose_main(
                [
                    "--scan-dir",
                    str(scan_dir),
                    "--runtime-index-path",
                    str(runtime_index_path),
                    "--task-board-path",
                    str(task_board_path),
                    "--proposal-policy",
                    str(proposal_policy_path),
                    "--proposal-index-path",
                    str(proposal_index_path),
                    "--proposals-dir",
                    str(proposals_dir),
                    "--materialize",
                ]
            )
            snapshot = build_queue_snapshot(scan_dir)
            write_queue_snapshot(queue_snapshot_path, snapshot)
            write_runtime_index(runtime_index_path, build_runtime_index(repo_root, scan_dir))
            write_json(task_board_path, build_task_board(repo_root, scan_dir))
            write_json(taskflow_index_path, reconcile_task_plans(scan_dir, repo_root=repo_root))
            if args.auto_agentic:
                for candidate in sorted(scan_dir.glob("*.json")):
                    card = load_experiment_card(candidate)
                    ensure_agentic_artifacts(
                        repo_root=repo_root,
                        card=card,
                        card_path=candidate,
                        execute_literature_search=False,
                    )
        selected = select_experiment_card(scan_dir)
        if selected is None:
            idle_cycles += 1
            write_daemon_status(
                status_path,
                {
                    "status": "idle",
                    "started_at_utc": started_at,
                    "updated_at_utc": utc_now_iso(),
                    "cycles": cycles,
                    "idle_cycles": idle_cycles,
                    "selected_card": "",
                    "ready_count": snapshot.get("ready_count", 0),
                    "controller_policy_path": str(policy_path.resolve()),
                },
            )
            if args.max_idle_cycles > 0 and idle_cycles >= int(args.max_idle_cycles):
                return 0
            if args.max_cycles > 0 and cycles >= int(args.max_cycles):
                return 0
            time.sleep(float(args.poll_interval_seconds))
            continue

        idle_cycles = 0
        selected_path = str(Path(selected).resolve())
        write_daemon_status(
            status_path,
            {
                "status": "running_card",
                "started_at_utc": started_at,
                "updated_at_utc": utc_now_iso(),
                "cycles": cycles,
                "idle_cycles": idle_cycles,
                "selected_card": selected_path,
                "ready_count": snapshot.get("ready_count", 0),
                "controller_policy_path": str(policy_path.resolve()),
            },
        )
        exit_code = run_research_queue_main(
            [
                "--experiment-card",
                selected_path,
                "--controller-policy",
                str(policy_path),
                "--human-approval",
                str(human_approval_path),
            ]
        )
        write_daemon_status(
            status_path,
            {
                "status": "post_card",
                "started_at_utc": started_at,
                "updated_at_utc": utc_now_iso(),
                "cycles": cycles,
                "idle_cycles": idle_cycles,
                "selected_card": selected_path,
                "exit_code": exit_code,
                "ready_count": build_queue_snapshot(scan_dir).get("ready_count", 0),
                "controller_policy_path": str(policy_path.resolve()),
            },
        )
        write_runtime_index(runtime_index_path, build_runtime_index(repo_root, scan_dir))
        write_json(task_board_path, build_task_board(repo_root, scan_dir))
        write_json(taskflow_index_path, reconcile_task_plans(scan_dir, repo_root=repo_root))
        if exit_code == 10:
            return exit_code
        if exit_code != 0 and not args.continue_after_failure:
            return exit_code
        if args.max_cycles > 0 and cycles >= int(args.max_cycles):
            return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
