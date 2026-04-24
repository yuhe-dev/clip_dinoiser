"""Runtime helpers for reproducible, resumable harness execution."""

from __future__ import annotations

import platform
import socket
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

from .contracts import ExperimentCard, RunManifest


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_utc_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def add_seconds_iso(value: str, seconds: float) -> str:
    return (parse_utc_iso(value) + timedelta(seconds=float(seconds))).isoformat()


def is_utc_iso_stale(value: str, *, ttl_seconds: float) -> bool:
    return datetime.now(timezone.utc) > parse_utc_iso(value) + timedelta(seconds=float(ttl_seconds))


def _run_git(repo_root: str | Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()
    except Exception:
        return ""


def collect_git_facts(repo_root: str | Path) -> Dict[str, Any]:
    sha = _run_git(repo_root, "rev-parse", "HEAD") or "unknown"
    branch = _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    status = _run_git(repo_root, "status", "--porcelain")
    return {
        "git_sha": sha,
        "git_branch": branch,
        "git_is_dirty": bool(status),
    }


def build_run_manifest(
    *,
    card: ExperimentCard,
    card_path: str | Path,
    repo_root: str | Path,
    output_dir: str | Path,
    judge_policy_path: str,
    invoked_command: str,
    started_at_utc: str,
    finished_at_utc: str,
    duration_seconds: float,
    metadata_overrides: Dict[str, Any] | None = None,
) -> RunManifest:
    git_facts = collect_git_facts(repo_root)
    metadata = {
        "budget_tier": card.budget_tier,
        "phase": card.phase,
        "owner": card.owner,
        "metric_name": card.metric_name,
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    return RunManifest(
        experiment_id=card.experiment_id,
        loop_kind=card.loop_kind,
        card_path=str(Path(card_path).resolve()),
        output_dir=str(Path(output_dir).resolve()),
        judge_policy_path=judge_policy_path,
        invoked_command=invoked_command,
        repo_root=str(Path(repo_root).resolve()),
        git_sha=str(git_facts["git_sha"]),
        git_branch=str(git_facts["git_branch"]),
        git_is_dirty=bool(git_facts["git_is_dirty"]),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        hostname=socket.gethostname(),
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        duration_seconds=duration_seconds,
        metadata=metadata,
    )
