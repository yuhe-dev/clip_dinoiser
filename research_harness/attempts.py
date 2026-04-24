"""Attempt-level manifests for repeated experiment execution."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict

from .contracts import AttemptManifest, ExperimentCard
from .registry import write_json
from .runtime import utc_now_iso


def allocate_attempt_id(experiment_id: str) -> str:
    stamp = utc_now_iso().replace(":", "-").replace("+00:00", "Z")
    return f"{experiment_id}_{stamp}_{os.getpid()}"


def attempt_dir_from_output(output_dir: str | Path, attempt_id: str) -> Path:
    return Path(output_dir) / "attempts" / attempt_id


def start_attempt(
    *,
    attempt_id: str,
    card: ExperimentCard,
    card_path: str | Path,
    session_id: str,
    output_dir: str | Path,
    runtime_profile_id: str,
    python_bin: str,
    metadata: Dict[str, Any] | None = None,
) -> Path:
    attempt_dir = attempt_dir_from_output(output_dir, attempt_id)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    manifest = AttemptManifest(
        attempt_id=attempt_id,
        experiment_id=card.experiment_id,
        session_id=session_id,
        card_path=str(Path(card_path).resolve()),
        output_dir=str(Path(output_dir).resolve()),
        attempt_dir=str(attempt_dir.resolve()),
        runtime_profile_id=str(runtime_profile_id),
        python_bin=str(python_bin),
        status="running",
        started_at_utc=utc_now_iso(),
        metadata=dict(metadata or {}),
    )
    write_json(attempt_dir / "attempt_manifest.json", manifest.to_dict())
    write_json(attempt_dir / "card_snapshot.json", card.to_dict())
    return attempt_dir


def finalize_attempt(
    *,
    attempt_dir: str | Path,
    attempt_id: str,
    card: ExperimentCard,
    card_path: str | Path,
    session_id: str,
    output_dir: str | Path,
    runtime_profile_id: str,
    python_bin: str,
    status: str,
    reason: str,
    exit_code: int,
    paths: Dict[str, str] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Path:
    manifest = AttemptManifest(
        attempt_id=attempt_id,
        experiment_id=card.experiment_id,
        session_id=session_id,
        card_path=str(Path(card_path).resolve()),
        output_dir=str(Path(output_dir).resolve()),
        attempt_dir=str(Path(attempt_dir).resolve()),
        runtime_profile_id=str(runtime_profile_id),
        python_bin=str(python_bin),
        status=status,
        started_at_utc="",
        finished_at_utc=utc_now_iso(),
        reason=str(reason),
        exit_code=int(exit_code),
        paths=dict(paths or {}),
        metadata=dict(metadata or {}),
    )
    existing = Path(attempt_dir) / "attempt_manifest.json"
    if existing.exists():
        payload = existing.read_text(encoding="utf-8")
        import json

        prior = json.loads(payload)
        manifest.started_at_utc = str(prior.get("started_at_utc", ""))
        if not manifest.metadata:
            manifest.metadata = dict(prior.get("metadata", {}))
    write_json(existing, manifest.to_dict())
    artifacts_dir = Path(attempt_dir) / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for label, raw_path in manifest.paths.items():
        candidate = Path(raw_path)
        if not candidate.exists() or not candidate.is_file():
            continue
        destination = artifacts_dir / f"{label}{candidate.suffix}"
        shutil.copy2(candidate, destination)
    return existing
