"""Detached supervisor for the long-running research daemon."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    ROOT = os.path.abspath(os.path.dirname(__file__))
    PARENT = os.path.dirname(ROOT)
    if PARENT not in sys.path:
        sys.path.insert(0, PARENT)

from clip_dinoiser.research_harness.controller import discover_repo_root
from clip_dinoiser.research_harness.registry import load_json, resolve_repo_path, write_json
from clip_dinoiser.research_harness.runtime import utc_now_iso


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start/stop/status detached research daemon.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--scan-dir", default=".slicetune/experiments")
    start.add_argument("--controller-policy", default=".slicetune/runtime/controller_policy.json")
    start.add_argument("--human-approval", default=".slicetune/approvals/human_review.json")
    start.add_argument("--poll-interval-seconds", type=float, default=30.0)
    start.add_argument("--continue-after-failure", action="store_true")
    start.add_argument("--auto-propose", action="store_true")
    start.add_argument("--auto-debate", action="store_true")
    start.add_argument("--auto-agentic", action="store_true")
    start.add_argument("--proposal-policy", default=".slicetune/runtime/proposal_policy.json")
    start.add_argument("--proposal-index-path", default=".slicetune/state/proposal_index.json")
    start.add_argument("--runtime-index-path", default=".slicetune/state/runtime_index.json")
    start.add_argument("--process-record-path", default="artifacts/research_harness/daemon/process_record.json")
    start.add_argument("--stdout-path", default="artifacts/research_harness/daemon/daemon.stdout.log")
    start.add_argument("--stderr-path", default="artifacts/research_harness/daemon/daemon.stderr.log")

    status = subparsers.add_parser("status")
    status.add_argument("--process-record-path", default="artifacts/research_harness/daemon/process_record.json")

    stop = subparsers.add_parser("stop")
    stop.add_argument("--process-record-path", default="artifacts/research_harness/daemon/process_record.json")
    return parser


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _load_process_record(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


def _write_process_record(path: Path, payload: dict) -> Path:
    write_json(path, payload)
    return path


def _status_payload(path: Path) -> dict:
    record = _load_process_record(path)
    pid = int(record.get("pid", 0)) if record else 0
    alive = _process_alive(pid)
    return {
        "record_path": str(path.resolve()),
        "exists": bool(record),
        "pid": pid,
        "alive": alive,
        "payload": record,
        "checked_at_utc": utc_now_iso(),
    }


def _start(args: argparse.Namespace) -> int:
    repo_root = discover_repo_root(Path(args.scan_dir))
    process_record_path = resolve_repo_path(repo_root, args.process_record_path)
    status = _status_payload(process_record_path)
    if status["alive"]:
        print(f"daemon_already_running: pid={status['pid']}")
        return 3

    stdout_path = resolve_repo_path(repo_root, args.stdout_path)
    stderr_path = resolve_repo_path(repo_root, args.stderr_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str((Path(repo_root) / "run_research_daemon.py").resolve()),
        "--scan-dir",
        str(resolve_repo_path(repo_root, args.scan_dir)),
        "--controller-policy",
        str(resolve_repo_path(repo_root, args.controller_policy)),
        "--human-approval",
        str(resolve_repo_path(repo_root, args.human_approval)),
        "--poll-interval-seconds",
        str(args.poll_interval_seconds),
        "--proposal-policy",
        str(resolve_repo_path(repo_root, args.proposal_policy)),
        "--proposal-index-path",
        str(resolve_repo_path(repo_root, args.proposal_index_path)),
        "--runtime-index-path",
        str(resolve_repo_path(repo_root, args.runtime_index_path)),
    ]
    if args.continue_after_failure:
        command.append("--continue-after-failure")
    if args.auto_propose:
        command.append("--auto-propose")
    if args.auto_debate:
        command.append("--auto-debate")
    if args.auto_agentic:
        command.append("--auto-agentic")

    with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open("a", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            stdout=stdout_handle,
            stderr=stderr_handle,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            text=True,
        )

    _write_process_record(
        process_record_path,
        {
            "pid": process.pid,
            "command": command,
            "repo_root": str(Path(repo_root).resolve()),
            "stdout_path": str(stdout_path.resolve()),
            "stderr_path": str(stderr_path.resolve()),
            "started_at_utc": utc_now_iso(),
        },
    )
    print(f"daemon_started: pid={process.pid}")
    return 0


def _stop(args: argparse.Namespace) -> int:
    repo_root = discover_repo_root(Path(args.process_record_path))
    process_record_path = resolve_repo_path(repo_root, args.process_record_path)
    record = _load_process_record(process_record_path)
    pid = int(record.get("pid", 0)) if record else 0
    if not _process_alive(pid):
        print("daemon_not_running")
        return 0
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    record["stopped_at_utc"] = utc_now_iso()
    record["status"] = "stopped"
    _write_process_record(process_record_path, record)
    print(f"daemon_stopped: pid={pid}")
    return 0


def _status(args: argparse.Namespace) -> int:
    repo_root = discover_repo_root(Path(args.process_record_path))
    process_record_path = resolve_repo_path(repo_root, args.process_record_path)
    payload = _status_payload(process_record_path)
    write_json(process_record_path.parent / "process_status.json", payload)
    print(f"daemon_status: alive={payload['alive']} pid={payload['pid']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "start":
        return _start(args)
    if args.command == "stop":
        return _stop(args)
    if args.command == "status":
        return _status(args)
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
