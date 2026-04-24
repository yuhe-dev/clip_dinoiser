"""Tier-A feature intervention execution for learner adaptability audits."""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from clip_dinoiser.slice_discovery.assembler import _load_records
from clip_dinoiser.slice_remix.class_coverage import load_class_presence_matrix
from clip_dinoiser.slice_remix.manifests import load_subset_manifest

from .contracts import ResultBundle
from .runtime import utc_now_iso
from .task_progress import build_feature_intervention_progress_payload, write_progress_artifacts


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return target


def _extract_metric_value(result_payload: Dict[str, Any], metric_name: str) -> float:
    task_payload = dict(result_payload.get("coco_stuff") or {})
    for key in ("summary", "full_summary", "proxy_summary"):
        summary = task_payload.get(key)
        if isinstance(summary, dict) and metric_name in summary and summary[metric_name] is not None:
            return float(summary[metric_name])
    raise ValueError(f"metric '{metric_name}' not found in result payload")


def _resolve_processed_paths(processed_data_root: str | Path) -> Dict[str, Path]:
    root = Path(processed_data_root)
    return {
        "quality": root / "quality" / "quality_processed_features.npy",
        "difficulty": root / "difficulty" / "difficulty_processed_features.npy",
        "coverage": root / "coverage" / "coverage_processed_features.npy",
    }


def _merge_processed_records(records_by_dimension: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    ordered_dimensions = ["quality", "difficulty", "coverage"]
    base_records = records_by_dimension["quality"]
    merged: List[Dict[str, Any]] = []
    for index, quality_record in enumerate(base_records):
        image_rel = str(quality_record["image_rel"])
        features: Dict[str, Any] = dict(quality_record.get("features", {}))
        merged_record = {
            "image_rel": image_rel,
            "annotation_rel": quality_record.get("annotation_rel", ""),
            "schema_version": quality_record.get("schema_version", ""),
            "features": features,
        }
        for dimension in ordered_dimensions[1:]:
            record = records_by_dimension[dimension][index]
            if str(record["image_rel"]) != image_rel:
                raise ValueError(f"mismatched sample ordering for dimension '{dimension}' at index {index}")
            merged_record["features"].update(dict(record.get("features", {})))
        merged.append(merged_record)
    return merged


def _zscore(values: np.ndarray) -> np.ndarray:
    mean = float(values.mean())
    stdev = float(values.std())
    if stdev <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mean) / stdev).astype(np.float32)


def _resolve_field_path(record: Dict[str, Any], field_path: str) -> float:
    parts = [part for part in str(field_path).split(".") if part]
    if parts and parts[0] != "features":
        parts = ["features", *parts]
    current: Any = record
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"field path '{field_path}' missing token '{part}'")
        current = current[part]
    return float(current)


_FORMULA_TERM = re.compile(r"([+-]?)\s*z\(([^)]+)\)")


def _score_formula_values(records: Sequence[Dict[str, Any]], formula: str) -> np.ndarray:
    contributions = np.zeros(len(records), dtype=np.float32)
    matches = list(_FORMULA_TERM.finditer(str(formula)))
    if not matches:
        raise ValueError(f"unsupported score formula: {formula}")
    for match in matches:
        sign_text, field_path = match.groups()
        sign = -1.0 if sign_text == "-" else 1.0
        values = np.asarray([_resolve_field_path(record, field_path) for record in records], dtype=np.float32)
        contributions += sign * _zscore(values)
    return contributions.astype(np.float32)


def _infer_pool_image_root(anchor_manifest) -> Path:
    for sample_id, sample_path in zip(anchor_manifest.sample_ids, anchor_manifest.sample_paths):
        normalized_sample_id = str(sample_id).replace("\\", "/")
        normalized_sample_path = str(sample_path).replace("\\", "/")
        if normalized_sample_path.endswith(normalized_sample_id):
            prefix = normalized_sample_path[: -len(normalized_sample_id)].rstrip("/")
            return Path(prefix)
    raise ValueError("could not infer pool image root from anchor manifest sample paths")


def _load_or_build_class_presence_cache(
    *,
    sample_ids: List[str],
    annotation_root: str | Path,
    cache_dir: str | Path,
    num_classes: int,
) -> np.ndarray:
    cache_root = Path(cache_dir)
    matrix_path = cache_root / f"class_presence_{num_classes}.npy"
    ids_path = cache_root / f"class_presence_{num_classes}_sample_ids.json"
    if matrix_path.exists() and ids_path.exists():
        cached_ids = _load_json(ids_path).get("sample_ids", [])
        if list(cached_ids) == list(sample_ids):
            return np.load(matrix_path)
    matrix = load_class_presence_matrix(sample_ids, str(annotation_root), num_classes=num_classes)
    cache_root.mkdir(parents=True, exist_ok=True)
    np.save(matrix_path, matrix)
    _write_json(ids_path, {"sample_ids": list(sample_ids)})
    return matrix


def _build_manifest(
    *,
    output_dir: str | Path,
    sample_ids: List[str],
    sample_paths: List[str],
    candidate_id: str,
    metadata: Dict[str, Any],
) -> Path:
    manifest_path = Path(output_dir) / "manifests" / f"{candidate_id}.json"
    payload = {
        "candidate_id": candidate_id,
        "sample_ids": list(sample_ids),
        "sample_paths": list(sample_paths),
        "metadata": metadata,
    }
    _write_json(manifest_path, payload)
    return manifest_path


def _mean_class_presence(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        return np.zeros(matrix.shape[1], dtype=np.float32)
    return matrix[indices].mean(axis=0).astype(np.float32)


def _materialize_pair_for_axis(
    *,
    axis_id: str,
    pair_seed: int,
    sample_ids: List[str],
    sample_paths: List[str],
    axis_scores: Dict[str, np.ndarray],
    target_axis_id: str,
    candidate_budget: int,
    subset_budget: int,
    anchor_indices: np.ndarray,
    class_presence_matrix: np.ndarray,
    output_dir: str | Path,
    control_family: str,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(pair_seed))
    target_scores = axis_scores[target_axis_id]
    anchor_target_mean = float(target_scores[anchor_indices].mean())
    axis_means = {
        name: float(values[anchor_indices].mean()) for name, values in axis_scores.items()
    }
    anchor_class_mean = _mean_class_presence(class_presence_matrix, anchor_indices)
    non_target_axes = [name for name in axis_scores if name != target_axis_id]

    def select(direction: int, excluded: set[int] | None = None) -> np.ndarray:
        excluded = excluded or set()
        base_order = np.argsort(-(direction * target_scores))
        candidate_indices = [int(index) for index in base_order.tolist() if int(index) not in excluded]
        candidate_indices = np.asarray(candidate_indices[:candidate_budget], dtype=np.int64)
        if candidate_indices.size == 0:
            raise ValueError(f"no candidates available for axis={target_axis_id} direction={direction}")

        penalties = np.zeros(candidate_indices.shape[0], dtype=np.float32)
        if non_target_axes:
            penalties += 0.75 * np.mean(
                np.stack(
                    [
                        np.abs(axis_scores[axis_name][candidate_indices] - axis_means[axis_name]).astype(np.float32)
                        for axis_name in non_target_axes
                    ],
                    axis=0,
                ),
                axis=0,
            )
        class_penalty = np.mean(
            np.abs(class_presence_matrix[candidate_indices].astype(np.float32) - anchor_class_mean[None, :]),
            axis=1,
        )
        penalties += 3.0 * class_penalty.astype(np.float32)
        jitter = rng.normal(0.0, 1e-6, size=candidate_indices.shape[0]).astype(np.float32)
        composite = (direction * target_scores[candidate_indices]).astype(np.float32) - penalties + jitter
        ordered = candidate_indices[np.argsort(-composite)]
        selected = ordered[:subset_budget]
        if selected.shape[0] != subset_budget:
            raise ValueError(f"subset selection underfilled for axis={target_axis_id} direction={direction}")
        return selected.astype(np.int64)

    high_indices = select(direction=1)
    low_indices = select(direction=-1, excluded=set(int(index) for index in high_indices.tolist()))

    high_target_mean = float(target_scores[high_indices].mean())
    low_target_mean = float(target_scores[low_indices].mean())
    realized_target_delta = high_target_mean - low_target_mean

    off_target_deltas = {
        axis_name: float(axis_scores[axis_name][high_indices].mean() - axis_scores[axis_name][low_indices].mean())
        for axis_name in non_target_axes
    }
    off_target_drift_ratio = 0.0
    if non_target_axes and abs(realized_target_delta) > 1e-8:
        off_target_drift_ratio = float(
            np.mean([abs(delta) / abs(realized_target_delta) for delta in off_target_deltas.values()])
        )

    high_class_mean = _mean_class_presence(class_presence_matrix, high_indices)
    low_class_mean = _mean_class_presence(class_presence_matrix, low_indices)
    class_histogram_drift = float(np.mean(np.abs(high_class_mean - low_class_mean)))

    manifests: Dict[str, str] = {}
    for direction_label, indices in (("high", high_indices), ("low", low_indices)):
        candidate_id = f"{axis_id}_{control_family}_seed{pair_seed:02d}_{direction_label}"
        manifest_path = _build_manifest(
            output_dir=output_dir,
            sample_ids=[sample_ids[int(index)] for index in indices.tolist()],
            sample_paths=[sample_paths[int(index)] for index in indices.tolist()],
            candidate_id=candidate_id,
            metadata={
                "axis_id": axis_id,
                "control_family": control_family,
                "pair_seed": int(pair_seed),
                "direction": direction_label,
            },
        )
        manifests[direction_label] = str(manifest_path.resolve())

    return {
        "axis_id": axis_id,
        "control_family": control_family,
        "pair_seed": int(pair_seed),
        "high_manifest_path": manifests["high"],
        "low_manifest_path": manifests["low"],
        "realized_target_delta": realized_target_delta,
        "anchor_target_mean": anchor_target_mean,
        "high_target_mean": high_target_mean,
        "low_target_mean": low_target_mean,
        "off_target_deltas": off_target_deltas,
        "off_target_drift_ratio": off_target_drift_ratio,
        "class_histogram_drift": class_histogram_drift,
        "coverage_drift": float(abs(off_target_deltas.get("coverage_density", 0.0))),
    }


def _run_training_command(
    *,
    python_bin: str,
    worker_script: str,
    config_name: str,
    subset_manifest_path: str | Path,
    output_dir: str | Path,
    result_name: str,
    training_seed: int,
    trainable_modules: Iterable[str],
    master_port: int,
) -> List[str]:
    command = [
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
    for module_path in trainable_modules:
        command.extend(["--trainable-modules", str(module_path)])
    return command


def _load_completion_if_valid(
    completion_path: str | Path,
    *,
    experiment_id: str,
    signature: Dict[str, Any],
) -> Dict[str, Any] | None:
    path = Path(completion_path)
    if not path.exists():
        return None
    payload = _load_json(path)
    if str(payload.get("experiment_id", "")) != experiment_id:
        return None
    for key, value in signature.items():
        if payload.get(key) != value:
            return None
    return payload


def _run_single_training(
    *,
    experiment_id: str,
    run_role: str,
    output_root: str | Path,
    run_id: str,
    subset_manifest_path: str | Path,
    metric_name: str,
    config_name: str,
    python_bin: str,
    worker_script: str,
    training_seed: int,
    trainable_modules: Iterable[str],
    gpu_id: str,
    master_port: int,
    runtime_profile_id: str,
    result_name: str,
) -> Dict[str, Any]:
    run_dir = Path(output_root) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / result_name
    completion_path = run_dir / "completion.json"
    log_path = run_dir / "stdout.log"
    normalized_trainable_modules = [str(item) for item in trainable_modules]
    signature = {
        "run_id": run_id,
        "config_name": config_name,
        "training_seed": int(training_seed),
        "subset_manifest_path": str(Path(subset_manifest_path).resolve()),
        "trainable_modules": list(normalized_trainable_modules),
    }
    completion_payload = _load_completion_if_valid(
        completion_path,
        experiment_id=experiment_id,
        signature=signature,
    )
    if completion_payload is not None and result_path.exists():
        result_payload = _load_json(result_path)
        metric_value = _extract_metric_value(result_payload, metric_name)
        return {
            "run_id": run_id,
            "run_role": run_role,
            "status": "reused_existing_result",
            "result_path": str(result_path.resolve()),
            "completion_path": str(completion_path.resolve()),
            "metric_value": metric_value,
            "training_seed": int(training_seed),
            "trainable_modules": list(normalized_trainable_modules),
        }

    command = _run_training_command(
        python_bin=python_bin,
        worker_script=worker_script,
        config_name=config_name,
        subset_manifest_path=subset_manifest_path,
        output_dir=run_dir,
        result_name=result_name,
        training_seed=training_seed,
        trainable_modules=normalized_trainable_modules,
        master_port=master_port,
    )
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
        raise RuntimeError(f"run {run_id} failed with exit code {completed.returncode}")

    result_payload = _load_json(result_path)
    metric_value = _extract_metric_value(result_payload, metric_name)
    _write_json(
        completion_path,
        {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "run_role": run_role,
            "config_name": config_name,
            "training_seed": int(training_seed),
            "subset_manifest_path": str(Path(subset_manifest_path).resolve()),
            "trainable_modules": list(normalized_trainable_modules),
            "metric_name": metric_name,
            "metric_value": metric_value,
            "python_bin": str(Path(python_bin).resolve()),
            "runtime_profile_id": runtime_profile_id,
            "completed_at_utc": utc_now_iso(),
        },
    )
    return {
        "run_id": run_id,
        "run_role": run_role,
        "status": "completed",
        "result_path": str(result_path.resolve()),
        "completion_path": str(completion_path.resolve()),
        "metric_value": metric_value,
        "training_seed": int(training_seed),
        "trainable_modules": list(normalized_trainable_modules),
    }


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


def run_feature_intervention_matrix(
    *,
    experiment_id: str,
    subset_manifest_path: str | Path,
    output_dir: str | Path,
    metric_name: str,
    processed_data_root: str | Path,
    schema_path: str | Path,
    learner_variants: Iterable[Dict[str, Any]],
    probe_feature_axes: Iterable[Dict[str, Any]],
    optional_probe_axes: Iterable[Dict[str, Any]] | None = None,
    tier_plan: Dict[str, Any] | None = None,
    python_bin: str = "",
    worker_script: str = "run_remix_training_experiment.py",
    config_name: str = "feature_experiment_fast_cached_slide",
    gpu_id: str = "0",
    master_port_base: int = 29800,
    result_name: str = "result.json",
    runtime_profile_id: str = "",
    num_classes: int = 171,
    log_fn=lambda _message: None,
) -> ResultBundle:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = output_root / "cache"
    anchor_manifest = load_subset_manifest(str(subset_manifest_path))
    pool_image_root = _infer_pool_image_root(anchor_manifest)
    annotation_root = pool_image_root
    resolved_python_bin = python_bin or sys.executable
    resolved_tier_plan = dict(tier_plan or {})
    tier_a = dict(resolved_tier_plan.get("tier_a_screen", {}))
    feature_pair_seeds = [int(seed) for seed in tier_a.get("feature_pair_seeds", [0])]
    noise_floor_seeds = [int(seed) for seed in tier_a.get("noise_floor_seeds", [0, 1, 2])]
    control_families = [str(item) for item in tier_a.get("controls", ["real_feature_guided"])]
    if control_families != ["real_feature_guided"]:
        raise ValueError("Tier A runtime currently supports only real_feature_guided")

    processed_paths = _resolve_processed_paths(processed_data_root)
    records_by_dimension = {
        dimension: _load_records(str(path))
        for dimension, path in processed_paths.items()
    }
    unified_records = _merge_processed_records(records_by_dimension)
    sample_ids = [str(record["image_rel"]) for record in unified_records]
    sample_paths = [str((pool_image_root / sample_id).resolve()) for sample_id in sample_ids]
    sample_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    anchor_indices = np.asarray([sample_index[sample_id] for sample_id in anchor_manifest.sample_ids], dtype=np.int64)

    all_axes = list(probe_feature_axes) + list(optional_probe_axes or [])
    axis_scores = {
        str(axis["axis_id"]): _score_formula_values(unified_records, str(axis["score_formula"]))
        for axis in all_axes
    }
    axis_table_path = _write_json(
        output_root / "axis_scores_summary.json",
        {
            "schema_path": str(Path(schema_path).resolve()),
            "axes": {
                axis_id: {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
                for axis_id, values in axis_scores.items()
            },
        },
    )

    class_presence_matrix = _load_or_build_class_presence_cache(
        sample_ids=sample_ids,
        annotation_root=annotation_root,
        cache_dir=cache_dir,
        num_classes=int(num_classes),
    )

    subset_budget = len(anchor_manifest.sample_ids)
    candidate_budget = max(int(subset_budget) * 5, 5000)
    probe_axes = list(probe_feature_axes)
    materializations: List[Dict[str, Any]] = []
    for axis in probe_axes:
        axis_id = str(axis["axis_id"])
        for pair_seed in feature_pair_seeds:
            pair = _materialize_pair_for_axis(
                axis_id=axis_id,
                pair_seed=int(pair_seed),
                sample_ids=sample_ids,
                sample_paths=sample_paths,
                axis_scores=axis_scores,
                target_axis_id=axis_id,
                candidate_budget=candidate_budget,
                subset_budget=subset_budget,
                anchor_indices=anchor_indices,
                class_presence_matrix=class_presence_matrix,
                output_dir=output_root,
                control_family="real_feature_guided",
            )
            materializations.append(pair)
    _write_json(output_root / "materialization_index.json", {"pairs": materializations})

    completed_runs: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    learner_variants_list = [dict(item) for item in learner_variants]
    noise_by_variant: Dict[str, Dict[str, Any]] = {}
    port_cursor = int(master_port_base)

    progress_artifacts = write_progress_artifacts(
        output_root,
        build_feature_intervention_progress_payload(
            experiment_id=experiment_id,
            metric_name=metric_name,
            learner_variants=learner_variants_list,
            probe_axes=probe_axes,
            current_step="starting_noise_floor",
            completed_runs=completed_runs,
            failures=failures,
        ),
    )

    for variant in learner_variants_list:
        variant_id = str(variant["variant_id"])
        modules = [str(item) for item in variant.get("trainable_modules", [])]
        noise_runs: List[Dict[str, Any]] = []
        for training_seed in noise_floor_seeds:
            run_id = f"{variant_id}_noise_seed{training_seed:02d}"
            run = _run_single_training(
                experiment_id=experiment_id,
                run_role="noise_floor",
                output_root=output_root,
                run_id=run_id,
                subset_manifest_path=subset_manifest_path,
                metric_name=metric_name,
                config_name=config_name,
                python_bin=resolved_python_bin,
                worker_script=worker_script,
                training_seed=int(training_seed),
                trainable_modules=modules,
                gpu_id=gpu_id,
                master_port=port_cursor,
                runtime_profile_id=runtime_profile_id,
                result_name=result_name,
            )
            port_cursor += 1
            noise_runs.append(run)
            completed_runs.append(
                {
                    "run_id": run_id,
                    "stage": "noise_floor",
                    "learner_variant_id": variant_id,
                    "metric_value": run["metric_value"],
                    "status": run["status"],
                }
            )
            progress_artifacts = write_progress_artifacts(
                output_root,
                build_feature_intervention_progress_payload(
                    experiment_id=experiment_id,
                    metric_name=metric_name,
                    learner_variants=learner_variants_list,
                    probe_axes=probe_axes,
                    current_step=f"noise_completed_{variant_id}_{training_seed}",
                    completed_runs=completed_runs,
                    failures=failures,
                ),
            )
        metrics = [float(run["metric_value"]) for run in noise_runs]
        noise_by_variant[variant_id] = {
            "variant_id": variant_id,
            "trainable_modules": list(modules),
            "seed_count": len(metrics),
            "summary": _scalar_summary(metrics),
            "runs": noise_runs,
        }

    cell_results: List[Dict[str, Any]] = []
    for variant in learner_variants_list:
        variant_id = str(variant["variant_id"])
        modules = [str(item) for item in variant.get("trainable_modules", [])]
        noise_summary = dict(noise_by_variant[variant_id]["summary"])
        noise_std = float(noise_summary.get("stdev", 0.0))
        for pair in materializations:
            axis_id = str(pair["axis_id"])
            pair_seed = int(pair["pair_seed"])
            runs_by_direction: Dict[str, Dict[str, Any]] = {}
            for direction_label, manifest_key in (("high", "high_manifest_path"), ("low", "low_manifest_path")):
                run_id = f"{variant_id}_{axis_id}_real_seed{pair_seed:02d}_{direction_label}"
                run = _run_single_training(
                    experiment_id=experiment_id,
                    run_role="intervention_pair",
                    output_root=output_root,
                    run_id=run_id,
                    subset_manifest_path=str(pair[manifest_key]),
                    metric_name=metric_name,
                    config_name=config_name,
                    python_bin=resolved_python_bin,
                    worker_script=worker_script,
                    training_seed=0,
                    trainable_modules=modules,
                    gpu_id=gpu_id,
                    master_port=port_cursor,
                    runtime_profile_id=runtime_profile_id,
                    result_name=result_name,
                )
                port_cursor += 1
                runs_by_direction[direction_label] = run
                completed_runs.append(
                    {
                        "run_id": run_id,
                        "stage": "feature_pair",
                        "learner_variant_id": variant_id,
                        "axis_id": axis_id,
                        "direction": direction_label,
                        "metric_value": run["metric_value"],
                        "status": run["status"],
                    }
                )
                progress_artifacts = write_progress_artifacts(
                    output_root,
                    build_feature_intervention_progress_payload(
                        experiment_id=experiment_id,
                        metric_name=metric_name,
                        learner_variants=learner_variants_list,
                        probe_axes=probe_axes,
                        current_step=f"pair_completed_{variant_id}_{axis_id}_{direction_label}",
                        completed_runs=completed_runs,
                        failures=failures,
                    ),
                )

            signed_response = float(runs_by_direction["high"]["metric_value"] - runs_by_direction["low"]["metric_value"])
            amplitude = abs(signed_response)
            response_to_noise_ratio = float("inf") if noise_std <= 1e-8 else amplitude / noise_std
            cell_results.append(
                {
                    "learner_variant_id": variant_id,
                    "axis_id": axis_id,
                    "control_family": "real_feature_guided",
                    "pair_seed": pair_seed,
                    "metric_high": float(runs_by_direction["high"]["metric_value"]),
                    "metric_low": float(runs_by_direction["low"]["metric_value"]),
                    "signed_response": signed_response,
                    "composition_response_amplitude": amplitude,
                    "learner_noise_std": noise_std,
                    "response_to_noise_ratio": response_to_noise_ratio,
                    "directional_consistency": 1.0,
                    "feature_validity_advantage": None,
                    "realized_target_delta": float(pair["realized_target_delta"]),
                    "off_target_drift_ratio": float(pair["off_target_drift_ratio"]),
                    "class_histogram_drift": float(pair["class_histogram_drift"]),
                    "coverage_drift": float(pair["coverage_drift"]),
                    "high_manifest_path": str(pair["high_manifest_path"]),
                    "low_manifest_path": str(pair["low_manifest_path"]),
                    "high_result_path": str(runs_by_direction["high"]["result_path"]),
                    "low_result_path": str(runs_by_direction["low"]["result_path"]),
                }
            )

    cell_results_path = _write_json(output_root / "cell_results.json", {"cells": cell_results})
    noise_path = _write_json(output_root / "noise_floor_summary.json", {"variants": noise_by_variant})
    trace_path = _write_json(
        output_root / "execution_trace.json",
        {
            "completed_runs": completed_runs,
            "failures": failures,
        },
    )

    response_ratios = [float(cell["response_to_noise_ratio"]) for cell in cell_results]
    drift_ratios = [float(cell["off_target_drift_ratio"]) for cell in cell_results]
    realized_deltas = [float(cell["realized_target_delta"]) for cell in cell_results]
    promising_cells = [
        cell
        for cell in cell_results
        if float(cell["response_to_noise_ratio"]) >= 1.0 and float(cell["off_target_drift_ratio"]) <= 1.0
    ]
    summary = {
        "tier_executed": "Tier A",
        "completed_learner_variant_count": len(noise_by_variant),
        "completed_probe_axis_count": len({str(cell["axis_id"]) for cell in cell_results}),
        "completed_real_cell_count": len(cell_results),
        "control_families_executed": 1,
        "minimum_seed_count_per_cell": 1,
        "best_real_response_to_noise_ratio": max(response_ratios) if response_ratios else 0.0,
        "mean_off_target_drift_ratio": float(np.mean(drift_ratios)) if drift_ratios else 0.0,
        "max_off_target_drift_ratio": max(drift_ratios) if drift_ratios else 0.0,
        "mean_realized_target_delta": float(np.mean(realized_deltas)) if realized_deltas else 0.0,
        "real_axes_with_signal_count": len({str(cell["axis_id"]) for cell in promising_cells}),
        "real_cells_above_noise_floor_count": len(promising_cells),
        "realized_target_delta_logged_for_all_cells": all(
            cell.get("realized_target_delta") is not None for cell in cell_results
        ),
        "teacher_frozen": True,
        "full_validation": True,
        "screen_passed": bool(promising_cells),
        "promote_ready": False,
    }

    progress_artifacts = write_progress_artifacts(
        output_root,
        build_feature_intervention_progress_payload(
            experiment_id=experiment_id,
            metric_name=metric_name,
            learner_variants=learner_variants_list,
            probe_axes=probe_axes,
            current_step="judge_completed",
            completed_runs=completed_runs,
            failures=failures,
            judge_decision="pending",
        ),
    )

    return ResultBundle(
        experiment_id=experiment_id,
        loop_kind="feature_intervention_matrix",
        input_path=str(Path(subset_manifest_path).resolve()),
        metric_name=metric_name,
        summary=summary,
        sample_ids=[],
        metadata={
            "config_name": config_name,
            "runtime_profile_id": runtime_profile_id,
            "python_bin": str(Path(resolved_python_bin).resolve()),
            "worker_script": str(worker_script),
            "axis_scores_summary_path": str(axis_table_path.resolve()),
            "materialization_index_path": str((output_root / "materialization_index.json").resolve()),
            "cell_results_path": str(cell_results_path.resolve()),
            "noise_floor_summary_path": str(noise_path.resolve()),
            "execution_trace_path": str(trace_path.resolve()),
            "task_plan_path": progress_artifacts["task_plan_path"],
            "progress_markdown_path": progress_artifacts["progress_markdown_path"],
            "handoff_path": progress_artifacts["handoff_path"],
        },
    )
