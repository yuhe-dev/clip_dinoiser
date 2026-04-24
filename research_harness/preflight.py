"""Worker-runtime preflight checks and runtime-profile selection."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from .contracts import ExperimentCard
from .registry import resolve_repo_path, write_json
from .runtime import utc_now_iso
from .runtime_profiles import load_runtime_profiles, resolve_runtime_profile_candidates


DEFAULT_REQUIRED_MODULES_BY_LOOP: Dict[str, List[str]] = {
    "same_subset_multi_seed": ["torch", "torchvision", "mmcv", "hydra", "omegaconf"],
    "learner_sensitivity_ladder": ["torch", "torchvision", "mmcv", "hydra", "omegaconf"],
    "feature_intervention_matrix": ["torch", "torchvision", "mmcv", "hydra", "omegaconf"],
}


def card_requires_worker_runtime(card: ExperimentCard) -> bool:
    if card.loop_kind in DEFAULT_REQUIRED_MODULES_BY_LOOP:
        return True
    metadata = dict(card.metadata)
    return bool(metadata.get("worker_script") or metadata.get("python_bin"))


def _dedupe(items: List[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def build_worker_requirements(card: ExperimentCard, repo_root: str | Path) -> Dict[str, Any]:
    metadata = dict(card.metadata)
    explicit_requirements = dict(metadata.get("worker_requirements", {}))
    required_modules = explicit_requirements.get("required_modules")
    if required_modules is None:
        required_modules = metadata.get("required_modules")
    if required_modules is None:
        required_modules = DEFAULT_REQUIRED_MODULES_BY_LOOP.get(card.loop_kind, [])

    default_worker_script = ""
    if card.loop_kind == "feature_intervention_matrix":
        default_worker_script = "run_remix_training_experiment.py"

    worker_script = str(
        explicit_requirements.get("worker_script")
        or metadata.get("worker_script", default_worker_script)
    ).strip()
    worker_script_path = str(resolve_repo_path(repo_root, worker_script).resolve()) if worker_script else ""

    default_config_name = ""
    if card.loop_kind == "feature_intervention_matrix":
        default_config_name = "feature_experiment_fast_cached_slide"
    config_name = str(explicit_requirements.get("config_name") or metadata.get("config_name", default_config_name)).strip()
    config_path = ""
    if config_name:
        config_candidate = resolve_repo_path(repo_root, f"configs/{config_name}.yaml")
        config_path = str(config_candidate.resolve())

    return {
        "required_modules": _dedupe([str(item) for item in required_modules]),
        "require_cuda": bool(
            explicit_requirements.get(
                "require_cuda",
                metadata.get(
                    "require_cuda",
                    card.loop_kind in {"same_subset_multi_seed", "learner_sensitivity_ladder", "feature_intervention_matrix"},
                ),
            )
        ),
        "worker_script": worker_script,
        "worker_script_path": worker_script_path,
        "gpu_id": str(explicit_requirements.get("gpu_id") or metadata.get("gpu_id", "0")).strip(),
        "config_name": config_name,
        "config_path": config_path,
    }


def probe_python_runtime(
    python_bin: str,
    *,
    required_modules: List[str],
    require_cuda: bool,
) -> Dict[str, Any]:
    target = Path(str(python_bin)).expanduser()
    payload: Dict[str, Any] = {
        "python_bin": str(target),
        "exists": target.exists(),
        "is_executable": os.access(target, os.X_OK),
        "required_modules": list(required_modules),
        "require_cuda": bool(require_cuda),
        "checked_at_utc": utc_now_iso(),
    }
    if not target.exists():
        payload.update(
            {
                "probe_succeeded": False,
                "passed": False,
                "reason": "python executable does not exist",
            }
        )
        return payload
    if not payload["is_executable"]:
        payload.update(
            {
                "probe_succeeded": False,
                "passed": False,
                "reason": "python executable exists but is not executable",
            }
        )
        return payload

    probe_script = """
import os
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
import importlib
import json
import sys

try:
    import numpy  # noqa: F401
except Exception:
    pass

required_modules = json.loads(sys.argv[1])
require_cuda = json.loads(sys.argv[2])
result = {
    "python_version": sys.version.split()[0],
    "sys_prefix": sys.prefix,
    "sys_executable": sys.executable,
    "modules": {},
    "missing_modules": [],
    "cuda_available": False,
    "cuda_device_count": 0,
}
torch_module = None
for name in required_modules:
    try:
        module = importlib.import_module(name)
        result["modules"][name] = {
            "ok": True,
            "version": str(getattr(module, "__version__", "")),
        }
        if name == "torch":
            torch_module = module
    except Exception as exc:
        result["modules"][name] = {"ok": False, "error": repr(exc)}
        result["missing_modules"].append(name)
if torch_module is None:
    try:
        import torch as torch_module
    except Exception:
        torch_module = None
if torch_module is not None:
    try:
        result["cuda_available"] = bool(torch_module.cuda.is_available())
        result["cuda_device_count"] = int(torch_module.cuda.device_count()) if result["cuda_available"] else 0
    except Exception as exc:
        result["cuda_available"] = False
        result["cuda_error"] = repr(exc)
result["passed"] = not result["missing_modules"] and (not require_cuda or result["cuda_available"])
print(json.dumps(result, ensure_ascii=False))
"""

    try:
        completed = subprocess.run(
            [
                str(target),
                "-c",
                probe_script,
                json.dumps(required_modules, ensure_ascii=False),
                json.dumps(bool(require_cuda)),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        payload.update(
            {
                "probe_succeeded": False,
                "passed": False,
                "reason": f"python probe could not start: {exc}",
            }
        )
        return payload
    payload["returncode"] = int(completed.returncode)
    payload["stdout"] = completed.stdout
    payload["stderr"] = completed.stderr

    if completed.returncode != 0:
        payload.update(
            {
                "probe_succeeded": False,
                "passed": False,
                "reason": "python probe exited non-zero",
            }
        )
        return payload

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        payload.update(
            {
                "probe_succeeded": False,
                "passed": False,
                "reason": "python probe returned no JSON payload",
            }
        )
        return payload

    try:
        probe_payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        payload.update(
            {
                "probe_succeeded": False,
                "passed": False,
                "reason": "python probe returned invalid JSON payload",
            }
        )
        return payload

    payload.update(probe_payload)
    payload["probe_succeeded"] = True
    payload["passed"] = bool(probe_payload.get("passed", False))
    if payload["passed"]:
        payload["reason"] = "preflight passed"
    elif payload.get("missing_modules"):
        payload["reason"] = f"missing modules: {', '.join(payload['missing_modules'])}"
    elif require_cuda and not payload.get("cuda_available", False):
        payload["reason"] = "CUDA is required but unavailable"
    else:
        payload["reason"] = "runtime preflight failed"
    return payload


def resolve_runtime_selection(
    card: ExperimentCard,
    *,
    repo_root: str | Path,
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    requirements = build_worker_requirements(card, repo_root)
    report: Dict[str, Any] = {
        "experiment_id": card.experiment_id,
        "loop_kind": card.loop_kind,
        "checked_at_utc": utc_now_iso(),
        "requires_worker_runtime": card_requires_worker_runtime(card),
        "worker_script": requirements["worker_script"],
        "worker_script_path": requirements["worker_script_path"],
        "worker_script_exists": bool(requirements["worker_script_path"])
        and Path(requirements["worker_script_path"]).exists(),
        "config_name": requirements["config_name"],
        "config_path": requirements["config_path"],
        "config_exists": bool(requirements["config_path"]) and Path(requirements["config_path"]).exists(),
        "required_modules": requirements["required_modules"],
        "require_cuda": requirements["require_cuda"],
        "gpu_id": requirements["gpu_id"],
        "candidate_reports": [],
        "selected_profile_id": "",
        "selected_python_bin": "",
        "status": "skipped",
        "reason": "loop does not require worker runtime",
    }
    if not report["requires_worker_runtime"]:
        return report

    if requirements["worker_script"] and not report["worker_script_exists"]:
        report["status"] = "blocked_preflight"
        report["reason"] = f"worker script does not exist: {requirements['worker_script_path']}"
        return report
    if requirements["config_name"] and not report["config_exists"]:
        report["status"] = "blocked_preflight"
        report["reason"] = f"config does not exist: {requirements['config_path']}"
        return report

    runtime_profiles_path = str(policy.get("runtime_profiles_path", "")).strip()
    registry: Dict[str, Any] = {"default_profile_candidates": [], "profiles": {}}
    if runtime_profiles_path:
        registry = load_runtime_profiles(str(resolve_repo_path(repo_root, runtime_profiles_path)))

    candidate_ids = resolve_runtime_profile_candidates(card, policy, registry)
    metadata_python = str(dict(card.metadata).get("python_bin", "")).strip()
    if not candidate_ids and metadata_python:
        candidate_ids = ["__card_python_bin__"]

    for profile_id in candidate_ids:
        if profile_id == "__card_python_bin__":
            profile = {
                "python_bin": metadata_python,
                "conda_env_name": "",
                "tags": ["card-metadata"],
                "enabled": True,
                "required_modules": [],
                "gpu_policy": "required" if requirements["require_cuda"] else "optional",
            }
        else:
            profile = dict(registry.get("profiles", {}).get(profile_id, {}))
        candidate_report: Dict[str, Any] = {
            "profile_id": profile_id,
            "profile_found": bool(profile),
            "enabled": bool(profile.get("enabled", True)),
            "conda_env_name": str(profile.get("conda_env_name", "")),
            "tags": list(profile.get("tags", [])),
            "python_bin": str(profile.get("python_bin", "")),
            "reasons": [],
        }
        if not candidate_report["profile_found"]:
            candidate_report["reasons"].append("runtime profile is not defined")
            report["candidate_reports"].append(candidate_report)
            continue
        if not candidate_report["enabled"]:
            candidate_report["reasons"].append("runtime profile is disabled")
            report["candidate_reports"].append(candidate_report)
            continue

        required_modules = _dedupe(
            list(requirements["required_modules"]) + [str(item) for item in profile.get("required_modules", [])]
        )
        probe = probe_python_runtime(
            candidate_report["python_bin"],
            required_modules=required_modules,
            require_cuda=requirements["require_cuda"],
        )
        candidate_report["probe"] = probe
        if requirements["require_cuda"]:
            device_count = int(probe.get("cuda_device_count", 0))
            try:
                gpu_index = int(str(requirements["gpu_id"]))
            except ValueError:
                gpu_index = 0
            if device_count <= gpu_index:
                probe["passed"] = False
                probe["reason"] = (
                    f"requested gpu_id={requirements['gpu_id']} but runtime reports only "
                    f"{device_count} visible CUDA device(s)"
                )
        if not probe.get("passed", False):
            candidate_report["reasons"].append(str(probe.get("reason", "runtime probe failed")))
            report["candidate_reports"].append(candidate_report)
            continue

        report["candidate_reports"].append(candidate_report)
        report["selected_profile_id"] = profile_id
        report["selected_python_bin"] = candidate_report["python_bin"]
        report["status"] = "passed"
        report["reason"] = f"selected runtime profile {profile_id}"
        return report

    report["status"] = "blocked_preflight"
    report["reason"] = "no runtime profile passed worker preflight"
    return report


def write_preflight_report(target_dir: str | Path, payload: Dict[str, Any]) -> Path:
    target = Path(target_dir) / "preflight_report.json"
    write_json(target, payload)
    return target
