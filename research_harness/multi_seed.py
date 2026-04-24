"""Same-subset multi-training-seed execution and summarization."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .contracts import ResultBundle
from .runtime import utc_now_iso
from .task_progress import build_same_subset_progress_payload, write_progress_artifacts


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


def _build_seed_manifest(
    base_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int,
) -> tuple[Path, str]:
    payload = _load_json(base_manifest_path)
    base_candidate_id = str(payload.get("candidate_id") or Path(base_manifest_path).stem)
    candidate_id = f"{base_candidate_id}_trainseed{int(seed):02d}"
    payload["candidate_id"] = candidate_id
    manifest_path = Path(output_dir) / "manifests" / f"{candidate_id}.json"
    _write_json(manifest_path, payload)
    return manifest_path, candidate_id


def _seed_command(
    *,
    python_bin: str,
    worker_script: str,
    config_name: str,
    subset_manifest_path: str | Path,
    output_dir: str | Path,
    result_name: str,
    seed: int,
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
        str(int(seed)),
    ]


def _write_progress(path: str | Path, payload: Dict[str, Any]) -> None:
    _write_json(path, payload)


def _write_task_progress(
    *,
    output_dir: str | Path,
    experiment_id: str,
    metric_name: str,
    training_seeds: Iterable[int],
    current_step: str,
    completed_runs: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> Dict[str, str]:
    payload = build_same_subset_progress_payload(
        experiment_id=experiment_id,
        metric_name=metric_name,
        training_seeds=training_seeds,
        current_step=current_step,
        completed_runs=completed_runs,
        failures=failures,
    )
    return write_progress_artifacts(output_dir, payload)


def _write_completion_sentinel(
    path: str | Path,
    *,
    experiment_id: str,
    candidate_id: str,
    training_seed: int,
    result_path: str | Path,
    metric_name: str,
    metric_value: float,
    command: List[str],
    python_bin: str,
    runtime_profile_id: str,
    config_name: str,
    subset_manifest_path: str | Path,
) -> None:
    _write_json(
        path,
        {
            "experiment_id": experiment_id,
            "candidate_id": candidate_id,
            "training_seed": int(training_seed),
            "result_path": str(Path(result_path).resolve()),
            "metric_name": metric_name,
            "metric_value": float(metric_value),
            "command": list(command),
            "python_bin": str(python_bin),
            "runtime_profile_id": str(runtime_profile_id),
            "config_name": str(config_name),
            "subset_manifest_path": str(Path(subset_manifest_path).resolve()),
            "completed_at_utc": utc_now_iso(),
        },
    )


def _sentinel_matches(
    completion_payload: Dict[str, Any],
    *,
    experiment_id: str,
    candidate_id: str,
    training_seed: int,
    metric_name: str,
    python_bin: str,
    runtime_profile_id: str,
    config_name: str,
    subset_manifest_path: str | Path,
) -> bool:
    recorded_python = str(completion_payload.get("python_bin", ""))
    if recorded_python:
        try:
            recorded_python = str(Path(recorded_python).resolve())
        except Exception:
            pass
    if str(completion_payload.get("experiment_id", "")) != experiment_id:
        return False
    if str(completion_payload.get("candidate_id", "")) != candidate_id:
        return False
    if int(completion_payload.get("training_seed", -1)) != int(training_seed):
        return False
    if str(completion_payload.get("metric_name", "")) != metric_name:
        return False
    if recorded_python != python_bin:
        return False

    # Backward compatibility: older sentinels did not record these fields.
    recorded_runtime = str(completion_payload.get("runtime_profile_id", ""))
    if recorded_runtime and recorded_runtime != str(runtime_profile_id):
        return False
    recorded_config = str(completion_payload.get("config_name", ""))
    if recorded_config and recorded_config != str(config_name):
        return False
    recorded_manifest = str(completion_payload.get("subset_manifest_path", ""))
    if recorded_manifest and recorded_manifest != str(Path(subset_manifest_path).resolve()):
        return False
    return True


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


def run_same_subset_multi_seed(
    *,
    experiment_id: str,
    subset_manifest_path: str | Path,
    output_dir: str | Path,
    metric_name: str,
    config_name: str,
    training_seeds: Iterable[int],
    python_bin: str = "",
    worker_script: str = "run_remix_training_experiment.py",
    gpu_id: str = "0",
    master_port_base: int = 29700,
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
    requested_seeds = [int(seed) for seed in training_seeds]
    completed_runs: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    base_manifest = _load_json(subset_manifest_path)
    base_candidate_id = str(base_manifest.get("candidate_id") or Path(subset_manifest_path).stem)
    python_exec = python_bin or sys.executable

    for index, seed in enumerate(requested_seeds):
        seed_manifest_path, candidate_id = _build_seed_manifest(
            subset_manifest_path,
            output_root,
            seed=seed,
        )
        run_dir = output_root / "runs" / candidate_id
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / result_name
        log_path = run_dir / "stdout.log"
        completion_path = run_dir / completion_filename

        command = _seed_command(
            python_bin=python_exec,
            worker_script=worker_script,
            config_name=config_name,
            subset_manifest_path=seed_manifest_path,
            output_dir=run_dir,
            result_name=result_name,
            seed=seed,
            master_port=int(master_port_base) + index,
        )
        if result_path.exists() and completion_path.exists():
            completion_payload = _load_json(completion_path)
            expected_python_bin = str(Path(python_exec).resolve())
            sentinel_matches = _sentinel_matches(
                completion_payload,
                experiment_id=experiment_id,
                candidate_id=candidate_id,
                training_seed=seed,
                metric_name=metric_name,
                python_bin=expected_python_bin,
                runtime_profile_id=str(runtime_profile_id),
                config_name=str(config_name),
                subset_manifest_path=seed_manifest_path,
            )
            if sentinel_matches:
                result_payload = _load_json(result_path)
                metric_value = _extract_metric_value(result_payload, metric_name)
                completed_runs.append(
                    {
                        "candidate_id": candidate_id,
                        "training_seed": seed,
                        "result_path": str(result_path.resolve()),
                        "metric_value": metric_value,
                        "status": "reused_existing_result",
                        "completion_path": str(completion_path.resolve()),
                    }
                )
                _write_progress(
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
                    training_seeds=requested_seeds,
                    current_step="reuse_existing_result",
                    completed_runs=completed_runs,
                    failures=failures,
                )
                continue
            log_fn(
                f"completion sentinel provenance mismatch; rerunning training_seed={seed} "
                f"candidate_id={candidate_id}"
            )
        if result_path.exists() and not completion_path.exists():
            log_fn(
                f"result exists without completion sentinel; rerunning training_seed={seed} "
                f"candidate_id={candidate_id}"
            )
        _write_progress(
            progress_path,
            {
                "experiment_id": experiment_id,
                "current_step": f"launch_seed_{seed}",
                "candidate_id": candidate_id,
                "command": command,
                "completed_runs": completed_runs,
                "failures": failures,
            },
        )
        _write_task_progress(
            output_dir=output_root,
            experiment_id=experiment_id,
            metric_name=metric_name,
            training_seeds=requested_seeds,
            current_step=f"launch_seed_{seed}",
            completed_runs=completed_runs,
            failures=failures,
        )
        log_fn(f"launch training_seed={seed} candidate_id={candidate_id}")
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
                "candidate_id": candidate_id,
                "training_seed": seed,
                "status": "failed",
                "exit_code": int(completed.returncode),
                "log_path": str(log_path.resolve()),
            }
            failures.append(failure)
            _write_progress(
                progress_path,
                {
                    "experiment_id": experiment_id,
                    "current_step": f"seed_failed_{seed}",
                    "completed_runs": completed_runs,
                    "failures": failures,
                },
            )
            _write_task_progress(
                output_dir=output_root,
                experiment_id=experiment_id,
                metric_name=metric_name,
                training_seeds=requested_seeds,
                current_step=f"seed_failed_{seed}",
                completed_runs=completed_runs,
                failures=failures,
            )
            if not continue_on_failure:
                raise RuntimeError(f"training seed {seed} failed with exit code {completed.returncode}")
            continue

        result_payload = _load_json(result_path)
        metric_value = _extract_metric_value(result_payload, metric_name)
        completed_runs.append(
            {
                "candidate_id": candidate_id,
                "training_seed": seed,
                "result_path": str(result_path.resolve()),
                "metric_value": metric_value,
                "status": "completed",
                "completion_path": str(completion_path.resolve()),
            }
        )
        _write_completion_sentinel(
            completion_path,
            experiment_id=experiment_id,
            candidate_id=candidate_id,
            training_seed=seed,
            result_path=result_path,
            metric_name=metric_name,
            metric_value=metric_value,
            command=command,
            python_bin=str(Path(python_exec).resolve()),
            runtime_profile_id=runtime_profile_id,
            config_name=config_name,
            subset_manifest_path=seed_manifest_path,
        )
        _write_progress(
            progress_path,
            {
                "experiment_id": experiment_id,
                "current_step": f"seed_completed_{seed}",
                "completed_runs": completed_runs,
                "failures": failures,
            },
        )
        _write_task_progress(
            output_dir=output_root,
            experiment_id=experiment_id,
            metric_name=metric_name,
            training_seeds=requested_seeds,
            current_step=f"seed_completed_{seed}",
            completed_runs=completed_runs,
            failures=failures,
        )

    metric_values = [float(item["metric_value"]) for item in completed_runs]
    if not metric_values:
        raise ValueError("no completed multi-seed runs were available to summarize")

    summary = {
        "base_candidate_id": base_candidate_id,
        "metric_name": metric_name,
        "requested_seed_count": len(requested_seeds),
        "completed_seed_count": len(completed_runs),
        "failure_count": len(failures),
        "training_seed_values": [int(item["training_seed"]) for item in completed_runs],
        "per_seed_metrics": {
            str(int(item["training_seed"])): float(item["metric_value"]) for item in completed_runs
        },
        **_scalar_summary(metric_values),
    }
    if failures:
        summary["failures"] = failures
    progress_artifacts = _write_task_progress(
        output_dir=output_root,
        experiment_id=experiment_id,
        metric_name=metric_name,
        training_seeds=requested_seeds,
        current_step="seed_execution_complete",
        completed_runs=completed_runs,
        failures=failures,
    )
    return ResultBundle(
        experiment_id=experiment_id,
        loop_kind="same_subset_multi_seed",
        input_path=str(Path(subset_manifest_path).resolve()),
        metric_name=metric_name,
        summary=summary,
        sample_ids=[str(base_candidate_id)],
        metadata={
            "config_name": config_name,
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
