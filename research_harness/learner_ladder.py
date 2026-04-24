"""Learner-sensitivity ladder execution and summarization."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .contracts import ResultBundle
from .runtime import utc_now_iso
from .task_progress import build_learner_ladder_progress_payload, write_progress_artifacts


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _extract_metric_value(result_payload: Dict[str, Any], metric_name: str) -> float:
    task_payload = dict(result_payload.get("coco_stuff") or {})
    for key in ("summary", "full_summary", "proxy_summary"):
        summary = task_payload.get(key)
        if isinstance(summary, dict) and metric_name in summary and summary[metric_name] is not None:
            return float(summary[metric_name])
    raise ValueError(f"metric '{metric_name}' not found in result payload")


def _build_regime_manifest(
    base_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    regime_id: str,
) -> tuple[Path, str]:
    payload = _load_json(base_manifest_path)
    base_candidate_id = str(payload.get("candidate_id") or Path(base_manifest_path).stem)
    candidate_id = f"{base_candidate_id}_{regime_id}"
    payload["candidate_id"] = candidate_id
    manifest_path = Path(output_dir) / "manifests" / f"{candidate_id}.json"
    _write_json(manifest_path, payload)
    return manifest_path, candidate_id


def _regime_command(
    *,
    python_bin: str,
    worker_script: str,
    config_name: str,
    subset_manifest_path: str | Path,
    output_dir: str | Path,
    result_name: str,
    training_seed: int,
    master_port: int,
) -> List[str]:
    return [
        str(python_bin),
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "--master_port",
        str(int(master_port)),
        str(worker_script),
        "--config",
        str(config_name),
        "--subset-manifest",
        str(subset_manifest_path),
        "--output-dir",
        str(output_dir),
        "--result-name",
        str(result_name),
        "--seed",
        str(int(training_seed)),
    ]


def _scalar_summary(values: List[float]) -> Dict[str, float]:
    ordered = sorted(values)
    count = len(values)
    mean = sum(values) / count
    variance = 0.0 if count == 1 else sum((value - mean) ** 2 for value in values) / (count - 1)
    stdev = variance ** 0.5
    return {
        "count": count,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "mean": mean,
        "median": ordered[count // 2] if count % 2 == 1 else (ordered[count // 2 - 1] + ordered[count // 2]) / 2.0,
        "stdev": stdev,
    }


def _write_task_progress(
    *,
    output_dir: str | Path,
    experiment_id: str,
    metric_name: str,
    regimes: List[Dict[str, Any]],
    current_step: str,
    completed_runs: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> Dict[str, str]:
    payload = build_learner_ladder_progress_payload(
        experiment_id=experiment_id,
        metric_name=metric_name,
        regimes=regimes,
        current_step=current_step,
        completed_runs=completed_runs,
        failures=failures,
    )
    return write_progress_artifacts(output_dir, payload)


def run_learner_sensitivity_ladder(
    *,
    experiment_id: str,
    subset_manifest_path: str | Path,
    output_dir: str | Path,
    metric_name: str,
    regimes: Iterable[Dict[str, Any]],
    python_bin: str = "",
    worker_script: str = "run_remix_training_experiment.py",
    gpu_id: str = "0",
    master_port_base: int = 29740,
    result_name: str = "result.json",
    continue_on_failure: bool = False,
    progress_filename: str = "progress.json",
    completion_filename: str = "completion.json",
    runtime_profile_id: str = "",
    log_fn=lambda _message: None,
) -> ResultBundle:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / progress_filename
    normalized_regimes: List[Dict[str, Any]] = []
    for index, raw in enumerate(regimes):
        regime = dict(raw)
        regime_id = str(regime.get("regime_id") or f"regime_{index:02d}")
        normalized_regimes.append(
            {
                "regime_id": regime_id,
                "config_name": str(regime["config_name"]),
                "training_seed": int(regime.get("training_seed", 0)),
            }
        )
    completed_runs: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    base_manifest = _load_json(subset_manifest_path)
    base_candidate_id = str(base_manifest.get("candidate_id") or Path(subset_manifest_path).stem)
    python_exec = python_bin or sys.executable

    for index, regime in enumerate(normalized_regimes):
        regime_id = str(regime["regime_id"])
        config_name = str(regime["config_name"])
        training_seed = int(regime.get("training_seed", 0))
        regime_manifest_path, candidate_id = _build_regime_manifest(
            subset_manifest_path,
            output_root,
            regime_id=regime_id,
        )
        run_dir = output_root / "runs" / candidate_id
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / result_name
        log_path = run_dir / "stdout.log"
        completion_path = run_dir / completion_filename
        command = _regime_command(
            python_bin=python_exec,
            worker_script=worker_script,
            config_name=config_name,
            subset_manifest_path=regime_manifest_path,
            output_dir=run_dir,
            result_name=result_name,
            training_seed=training_seed,
            master_port=int(master_port_base) + index,
        )
        if result_path.exists() and completion_path.exists():
            completion_payload = _load_json(completion_path)
            if (
                str(completion_payload.get("experiment_id", "")) == experiment_id
                and str(completion_payload.get("regime_id", "")) == regime_id
                and str(completion_payload.get("config_name", "")) == config_name
                and int(completion_payload.get("training_seed", -1)) == training_seed
            ):
                result_payload = _load_json(result_path)
                metric_value = _extract_metric_value(result_payload, metric_name)
                completed_runs.append(
                    {
                        "regime_id": regime_id,
                        "candidate_id": candidate_id,
                        "config_name": config_name,
                        "training_seed": training_seed,
                        "result_path": str(result_path.resolve()),
                        "metric_value": metric_value,
                        "status": "reused_existing_result",
                    }
                )
                _write_json(
                    progress_path,
                    {
                        "experiment_id": experiment_id,
                        "current_step": "reuse_existing_result",
                        "completed_runs": completed_runs,
                        "failures": failures,
                    },
                )
                _write_task_progress(
                    output_dir=output_root,
                    experiment_id=experiment_id,
                    metric_name=metric_name,
                    regimes=normalized_regimes,
                    current_step="reuse_existing_result",
                    completed_runs=completed_runs,
                    failures=failures,
                )
                continue

        _write_json(
            progress_path,
            {
                "experiment_id": experiment_id,
                "current_step": f"launch_regime_{regime_id}",
                "completed_runs": completed_runs,
                "failures": failures,
            },
        )
        _write_task_progress(
            output_dir=output_root,
            experiment_id=experiment_id,
            metric_name=metric_name,
            regimes=normalized_regimes,
            current_step=f"launch_regime_{regime_id}",
            completed_runs=completed_runs,
            failures=failures,
        )
        log_fn(f"launch regime_id={regime_id} config={config_name} training_seed={training_seed}")
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=os.getcwd(),
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": str(gpu_id),
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                },
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            failure = {
                "regime_id": regime_id,
                "config_name": config_name,
                "training_seed": training_seed,
                "status": "failed",
                "exit_code": int(completed.returncode),
                "log_path": str(log_path.resolve()),
            }
            failures.append(failure)
            _write_json(
                progress_path,
                {
                    "experiment_id": experiment_id,
                    "current_step": f"regime_failed_{regime_id}",
                    "completed_runs": completed_runs,
                    "failures": failures,
                },
            )
            _write_task_progress(
                output_dir=output_root,
                experiment_id=experiment_id,
                metric_name=metric_name,
                regimes=normalized_regimes,
                current_step=f"regime_failed_{regime_id}",
                completed_runs=completed_runs,
                failures=failures,
            )
            if not continue_on_failure:
                raise RuntimeError(f"learner regime {regime_id} failed with exit code {completed.returncode}")
            continue

        result_payload = _load_json(result_path)
        metric_value = _extract_metric_value(result_payload, metric_name)
        completed_runs.append(
            {
                "regime_id": regime_id,
                "candidate_id": candidate_id,
                "config_name": config_name,
                "training_seed": training_seed,
                "result_path": str(result_path.resolve()),
                "metric_value": metric_value,
                "status": "completed",
            }
        )
        _write_json(
            completion_path,
            {
                "experiment_id": experiment_id,
                "regime_id": regime_id,
                "candidate_id": candidate_id,
                "config_name": config_name,
                "training_seed": training_seed,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "python_bin": str(Path(python_exec).resolve()),
                "runtime_profile_id": runtime_profile_id,
                "completed_at_utc": utc_now_iso(),
            },
        )
        _write_json(
            progress_path,
            {
                "experiment_id": experiment_id,
                "current_step": f"regime_completed_{regime_id}",
                "completed_runs": completed_runs,
                "failures": failures,
            },
        )
        _write_task_progress(
            output_dir=output_root,
            experiment_id=experiment_id,
            metric_name=metric_name,
            regimes=normalized_regimes,
            current_step=f"regime_completed_{regime_id}",
            completed_runs=completed_runs,
            failures=failures,
        )

    metric_values = [float(item["metric_value"]) for item in completed_runs]
    if not metric_values:
        raise ValueError("no completed learner-sensitivity runs were available to summarize")
    per_regime_metrics = {
        str(item["regime_id"]): float(item["metric_value"])
        for item in completed_runs
    }
    best_run = max(completed_runs, key=lambda item: float(item["metric_value"]))
    worst_run = min(completed_runs, key=lambda item: float(item["metric_value"]))
    baseline_run = completed_runs[0]
    summary = {
        "base_candidate_id": base_candidate_id,
        "metric_name": metric_name,
        "requested_regime_count": len(normalized_regimes),
        "completed_regime_count": len(completed_runs),
        "failure_count": len(failures),
        "regime_ids": [str(item["regime_id"]) for item in completed_runs],
        "per_regime_metrics": per_regime_metrics,
        "best_regime_id": str(best_run["regime_id"]),
        "best_config_name": str(best_run["config_name"]),
        "worst_regime_id": str(worst_run["regime_id"]),
        "baseline_regime_id": str(baseline_run["regime_id"]),
        "best_minus_baseline": float(best_run["metric_value"]) - float(baseline_run["metric_value"]),
        "regime_range": max(metric_values) - min(metric_values),
        **_scalar_summary(metric_values),
    }
    if failures:
        summary["failures"] = failures
    progress_artifacts = _write_task_progress(
        output_dir=output_root,
        experiment_id=experiment_id,
        metric_name=metric_name,
        regimes=normalized_regimes,
        current_step="regime_execution_complete",
        completed_runs=completed_runs,
        failures=failures,
    )
    return ResultBundle(
        experiment_id=experiment_id,
        loop_kind="learner_sensitivity_ladder",
        input_path=str(Path(subset_manifest_path).resolve()),
        metric_name=metric_name,
        summary=summary,
        sample_ids=[str(base_candidate_id)],
        metadata={
            "regimes": normalized_regimes,
            "progress_path": str(progress_path.resolve()),
            "task_plan_path": progress_artifacts["task_plan_path"],
            "progress_markdown_path": progress_artifacts["progress_markdown_path"],
            "handoff_path": progress_artifacts["handoff_path"],
            "runs_root": str((output_root / "runs").resolve()),
            "worker_python_bin": str(Path(python_exec).resolve()),
            "runtime_profile_id": runtime_profile_id,
            "completion_filename": completion_filename,
        },
    )
