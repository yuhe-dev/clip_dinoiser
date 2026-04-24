"""Single-step CLI entrypoint for the thin, agent-centered research harness."""

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

from clip_dinoiser.research_harness.judge import judge_noise_floor
from clip_dinoiser.research_harness.judge import judge_same_subset_multi_seed
from clip_dinoiser.research_harness.judge import judge_literature_radar
from clip_dinoiser.research_harness.judge import judge_learner_sensitivity_ladder
from clip_dinoiser.research_harness.judge import judge_feature_intervention_matrix
from clip_dinoiser.research_harness.agentic import ensure_agentic_artifacts
from clip_dinoiser.research_harness.agentic_judge import apply_agentic_judge, write_judgment_artifacts
from clip_dinoiser.research_harness.contracts import ResultBundle
from clip_dinoiser.research_harness.feature_intervention import run_feature_intervention_matrix
from clip_dinoiser.research_harness.literature import summarize_literature_results
from clip_dinoiser.research_harness.learner_ladder import run_learner_sensitivity_ladder
from clip_dinoiser.research_harness.multi_seed import run_same_subset_multi_seed
from clip_dinoiser.research_harness.noise_floor import build_noise_floor_bundle
from clip_dinoiser.research_harness.preflight import build_worker_requirements, probe_python_runtime
from clip_dinoiser.research_harness.registry import (
    load_experiment_card,
    load_json,
    load_judge_policy,
    resolve_repo_path,
    write_json,
    write_judge_report_json,
    write_judge_report_markdown,
    write_run_manifest,
    write_result_bundle,
)
from clip_dinoiser.research_harness.runtime import build_run_manifest, utc_now_iso
from clip_dinoiser.research_harness.task_progress import (
    build_learner_ladder_progress_payload,
    build_feature_intervention_progress_payload,
    build_literature_progress_payload,
    build_noise_floor_progress_payload,
    build_same_subset_progress_payload,
    write_progress_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one research-harness tick.")
    parser.add_argument(
        "--experiment-card",
        required=True,
        help="Path to a machine-readable experiment card JSON file.",
    )
    parser.add_argument("--python-bin-override", default="", help="Override worker python executable for this tick.")
    parser.add_argument("--runtime-profile-id", default="", help="Resolved runtime profile id for this tick.")
    return parser


def _discover_repo_root(card_path: Path) -> Path:
    resolved = card_path.resolve()
    for candidate in [resolved.parent, *resolved.parents]:
        if (candidate / ".slicetune").is_dir() and (candidate / "AGENTS.md").exists():
            return candidate
    return resolved.parent


def _resolve_judge_policy(card, repo_root: Path) -> tuple[dict, str]:
    if card.judge_policy_path:
        policy_path = resolve_repo_path(repo_root, card.judge_policy_path)
        return load_judge_policy(policy_path), str(policy_path.resolve())
    return dict(card.judge_config), "<legacy:embedded-judge-config>"


def _prime_agentic_artifacts(*, repo_root: Path, card, card_path: Path, execute_literature_search: bool = False) -> None:
    ensure_agentic_artifacts(
        repo_root=repo_root,
        card=card,
        card_path=card_path,
        execute_literature_search=execute_literature_search,
    )


def _load_agentic_overlay(output_dir: Path) -> tuple[dict, dict]:
    rubric = {}
    context_packet = {}
    rubric_path = output_dir / "agentic" / "evaluation_rubric.json"
    context_path = output_dir / "agentic" / "context_snapshot.json"
    if rubric_path.exists():
        rubric = load_json(rubric_path)
    if context_path.exists():
        context_packet = load_json(context_path)
    return rubric, context_packet


def _run_same_subset_multi_seed(
    card_path: Path,
    *,
    python_bin_override: str = "",
    runtime_profile_id: str = "",
) -> int:
    repo_root = _discover_repo_root(card_path)
    card = load_experiment_card(card_path)
    input_path = resolve_repo_path(repo_root, card.input_path)
    output_dir = resolve_repo_path(repo_root, card.output_dir)
    judge_policy, judge_policy_path = _resolve_judge_policy(card, repo_root)
    metadata = dict(card.metadata)
    started_at = utc_now_iso()
    started_perf = time.perf_counter()
    _prime_agentic_artifacts(repo_root=repo_root, card=card, card_path=card_path)

    training_seeds = metadata.get("training_seeds", [0, 1, 2])
    resolved_python_bin = str(python_bin_override or metadata.get("python_bin", sys.executable))
    requirements = build_worker_requirements(card, repo_root)
    preflight_report = probe_python_runtime(
        resolved_python_bin,
        required_modules=list(requirements["required_modules"]),
        require_cuda=bool(requirements["require_cuda"]),
    )
    preflight_report.update(
        {
            "runtime_profile_id": runtime_profile_id,
            "worker_script": requirements["worker_script"],
            "worker_script_path": requirements["worker_script_path"],
            "config_name": requirements["config_name"],
            "config_path": requirements["config_path"],
        }
    )
    write_json(output_dir / "preflight_report.json", preflight_report)
    if not preflight_report.get("passed", False):
        raise RuntimeError(f"worker runtime preflight failed: {preflight_report.get('reason', 'unknown reason')}")

    bundle = run_same_subset_multi_seed(
        experiment_id=card.experiment_id,
        subset_manifest_path=input_path,
        output_dir=output_dir,
        metric_name=card.metric_name,
        config_name=str(metadata.get("config_name", "feature_experiment_fast_cached_slide")),
        training_seeds=[int(seed) for seed in training_seeds],
        python_bin=resolved_python_bin,
        worker_script=str(metadata.get("worker_script", "run_remix_training_experiment.py")),
        gpu_id=str(metadata.get("gpu_id", "0")),
        master_port_base=int(metadata.get("master_port_base", 29700)),
        result_name=str(metadata.get("result_name", "result.json")),
        continue_on_failure=bool(metadata.get("continue_on_failure", False)),
        completion_filename=str(metadata.get("completion_filename", "completion.json")),
        runtime_profile_id=runtime_profile_id,
        log_fn=lambda message: print(f"[research_tick] {message}", flush=True),
    )
    report = judge_same_subset_multi_seed(bundle, **judge_policy)
    rubric, context_packet = _load_agentic_overlay(output_dir)
    report, judgment_brief = apply_agentic_judge(
        card=card,
        context_packet=context_packet,
        bundle=bundle,
        mechanical_report=report,
        evaluation_rubric=rubric,
    )

    progress_path = output_dir / "progress.json"
    progress_payload = {}
    if progress_path.exists():
        progress_payload = load_json(progress_path)
    progress_artifacts = write_progress_artifacts(
        output_dir,
        build_same_subset_progress_payload(
            experiment_id=card.experiment_id,
            metric_name=card.metric_name,
            training_seeds=[int(seed) for seed in training_seeds],
            current_step="judge_completed",
            completed_runs=list(progress_payload.get("completed_runs", [])),
            failures=list(progress_payload.get("failures", [])),
            judge_decision=str(report.decision),
            human_review_required=bool(card.human_review_required),
        ),
    )
    bundle.metadata.update(
        {
            "task_plan_path": progress_artifacts["task_plan_path"],
            "progress_markdown_path": progress_artifacts["progress_markdown_path"],
            "handoff_path": progress_artifacts["handoff_path"],
        }
    )

    write_result_bundle(output_dir, bundle)
    write_judge_report_json(output_dir, report)
    write_judge_report_markdown(output_dir, report)
    write_judgment_artifacts(output_dir, judgment_brief)
    finished_at = utc_now_iso()
    duration_seconds = time.perf_counter() - started_perf
    manifest = build_run_manifest(
        card=card,
        card_path=card_path,
        repo_root=repo_root,
        output_dir=output_dir,
        judge_policy_path=judge_policy_path,
        invoked_command=" ".join(sys.argv) if sys.argv else "run_research_tick",
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_seconds=duration_seconds,
        metadata_overrides={
            "worker_runtime_profile_id": runtime_profile_id,
            "worker_python_bin": resolved_python_bin,
            "worker_preflight_path": str((output_dir / "preflight_report.json").resolve()),
        },
    )
    write_run_manifest(output_dir, manifest)
    _prime_agentic_artifacts(repo_root=repo_root, card=card, card_path=card_path)

    print(
        f"{card.experiment_id}: decision={report.decision} "
        f"completed={bundle.summary['completed_seed_count']} "
        f"mean={bundle.summary['mean']:.4f} std={bundle.summary['stdev']:.4f}"
    )
    return 0


def _run_noise_floor(card_path: Path) -> int:
    repo_root = _discover_repo_root(card_path)
    card = load_experiment_card(card_path)
    input_path = resolve_repo_path(repo_root, card.input_path)
    output_dir = resolve_repo_path(repo_root, card.output_dir)
    judge_policy, judge_policy_path = _resolve_judge_policy(card, repo_root)
    started_at = utc_now_iso()
    started_perf = time.perf_counter()
    _prime_agentic_artifacts(repo_root=repo_root, card=card, card_path=card_path)

    bundle = build_noise_floor_bundle(
        experiment_id=card.experiment_id,
        input_path=input_path,
        metric_name=card.metric_name,
        metadata=card.metadata,
    )
    report = judge_noise_floor(bundle, **judge_policy)
    rubric, context_packet = _load_agentic_overlay(output_dir)
    report, judgment_brief = apply_agentic_judge(
        card=card,
        context_packet=context_packet,
        bundle=bundle,
        mechanical_report=report,
        evaluation_rubric=rubric,
    )

    progress_artifacts = write_progress_artifacts(
        output_dir,
        build_noise_floor_progress_payload(
            experiment_id=card.experiment_id,
            metric_name=card.metric_name,
            summary=bundle.summary,
            judge_decision=str(report.decision),
            human_review_required=bool(card.human_review_required),
        ),
    )
    bundle.metadata.update(
        {
            "task_plan_path": progress_artifacts["task_plan_path"],
            "progress_markdown_path": progress_artifacts["progress_markdown_path"],
            "handoff_path": progress_artifacts["handoff_path"],
        }
    )

    write_result_bundle(output_dir, bundle)
    write_judge_report_json(output_dir, report)
    write_judge_report_markdown(output_dir, report)
    write_judgment_artifacts(output_dir, judgment_brief)
    finished_at = utc_now_iso()
    duration_seconds = time.perf_counter() - started_perf
    manifest = build_run_manifest(
        card=card,
        card_path=card_path,
        repo_root=repo_root,
        output_dir=output_dir,
        judge_policy_path=judge_policy_path,
        invoked_command=" ".join(sys.argv) if sys.argv else "run_research_tick",
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_seconds=duration_seconds,
    )
    write_run_manifest(output_dir, manifest)
    _prime_agentic_artifacts(repo_root=repo_root, card=card, card_path=card_path)

    print(
        f"{card.experiment_id}: decision={report.decision} "
        f"count={bundle.summary['count']} mean={bundle.summary['mean']:.4f} "
        f"std={bundle.summary['stdev']:.4f} range={bundle.summary['range']:.4f}"
    )
    return 0


def _run_literature_radar(card_path: Path) -> int:
    repo_root = _discover_repo_root(card_path)
    card = load_experiment_card(card_path)
    output_dir = resolve_repo_path(repo_root, card.output_dir)
    judge_policy, judge_policy_path = _resolve_judge_policy(card, repo_root)
    started_at = utc_now_iso()
    started_perf = time.perf_counter()

    _prime_agentic_artifacts(
        repo_root=repo_root,
        card=card,
        card_path=card_path,
        execute_literature_search=True,
    )

    query_plan = load_json(output_dir / "agentic" / "literature_query_plan.json")
    search_report = load_json(output_dir / "agentic" / "literature_search_report.json")
    method_cards_path = output_dir / "agentic" / "method_cards.jsonl"
    ranked_results = []
    if method_cards_path.exists():
        with method_cards_path.open("r", encoding="utf-8") as handle:
            import json

            ranked_results = [json.loads(line) for line in handle if line.strip()]
    summary = summarize_literature_results(
        query_plan,
        ranked_results,
        search_errors=dict(search_report.get("search_errors", {})),
    )
    bundle = ResultBundle(
        experiment_id=card.experiment_id,
        loop_kind=card.loop_kind,
        input_path=card.input_path or str((output_dir / "agentic" / "literature_query_plan.json").resolve()),
        metric_name=card.metric_name or "literature_score",
        summary=summary,
        sample_ids=[str(item.get("work_id", "")) for item in ranked_results[:20]],
        metadata={
            "query_plan_path": str((output_dir / "agentic" / "literature_query_plan.json").resolve()),
            "search_report_path": str((output_dir / "agentic" / "literature_search_report.json").resolve()),
            "method_cards_path": str(method_cards_path.resolve()),
            "top_titles": list(summary.get("top_titles", [])),
        },
    )
    report = judge_literature_radar(bundle, **judge_policy)
    rubric, context_packet = _load_agentic_overlay(output_dir)
    report, judgment_brief = apply_agentic_judge(
        card=card,
        context_packet=context_packet,
        bundle=bundle,
        mechanical_report=report,
        evaluation_rubric=rubric,
    )

    progress_artifacts = write_progress_artifacts(
        output_dir,
        build_literature_progress_payload(
            experiment_id=card.experiment_id,
            current_step="judge_completed",
            query_count=int(summary.get("query_count", 0)),
            ranked_result_count=int(summary.get("ranked_result_count", 0)),
            reproduce_count=int(summary.get("reproduce_count", 0)),
            search_error_count=int(summary.get("search_error_count", 0)),
            judge_decision=str(report.decision),
            human_review_required=bool(card.human_review_required),
        ),
    )
    bundle.metadata.update(
        {
            "task_plan_path": progress_artifacts["task_plan_path"],
            "progress_markdown_path": progress_artifacts["progress_markdown_path"],
            "handoff_path": progress_artifacts["handoff_path"],
        }
    )

    write_result_bundle(output_dir, bundle)
    write_judge_report_json(output_dir, report)
    write_judge_report_markdown(output_dir, report)
    write_judgment_artifacts(output_dir, judgment_brief)
    finished_at = utc_now_iso()
    duration_seconds = time.perf_counter() - started_perf
    manifest = build_run_manifest(
        card=card,
        card_path=card_path,
        repo_root=repo_root,
        output_dir=output_dir,
        judge_policy_path=judge_policy_path,
        invoked_command=" ".join(sys.argv) if sys.argv else "run_research_tick",
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_seconds=duration_seconds,
    )
    write_run_manifest(output_dir, manifest)
    _prime_agentic_artifacts(repo_root=repo_root, card=card, card_path=card_path)

    print(
        f"{card.experiment_id}: decision={report.decision} "
        f"queries={summary['query_count']} ranked={summary['ranked_result_count']} "
        f"reproduce={summary['reproduce_count']} errors={summary['search_error_count']}"
    )
    return 0


def _run_learner_sensitivity_ladder(
    card_path: Path,
    *,
    python_bin_override: str = "",
    runtime_profile_id: str = "",
) -> int:
    repo_root = _discover_repo_root(card_path)
    card = load_experiment_card(card_path)
    input_path = resolve_repo_path(repo_root, card.input_path)
    output_dir = resolve_repo_path(repo_root, card.output_dir)
    judge_policy, judge_policy_path = _resolve_judge_policy(card, repo_root)
    metadata = dict(card.metadata)
    started_at = utc_now_iso()
    started_perf = time.perf_counter()

    _prime_agentic_artifacts(repo_root=repo_root, card=card, card_path=card_path)

    resolved_python_bin = str(python_bin_override or metadata.get("python_bin", sys.executable))
    requirements = build_worker_requirements(card, repo_root)
    preflight_report = probe_python_runtime(
        resolved_python_bin,
        required_modules=list(requirements["required_modules"]),
        require_cuda=bool(requirements["require_cuda"]),
    )
    preflight_report.update(
        {
            "runtime_profile_id": runtime_profile_id,
            "worker_script": requirements["worker_script"],
            "worker_script_path": requirements["worker_script_path"],
        }
    )
    write_json(output_dir / "preflight_report.json", preflight_report)
    if not preflight_report.get("passed", False):
        raise RuntimeError(f"worker runtime preflight failed: {preflight_report.get('reason', 'unknown reason')}")

    regimes = list(metadata.get("regimes", []))
    bundle = run_learner_sensitivity_ladder(
        experiment_id=card.experiment_id,
        subset_manifest_path=input_path,
        output_dir=output_dir,
        metric_name=card.metric_name,
        regimes=regimes,
        python_bin=resolved_python_bin,
        worker_script=str(metadata.get("worker_script", "run_remix_training_experiment.py")),
        gpu_id=str(metadata.get("gpu_id", "0")),
        master_port_base=int(metadata.get("master_port_base", 29740)),
        result_name=str(metadata.get("result_name", "result.json")),
        continue_on_failure=bool(metadata.get("continue_on_failure", False)),
        runtime_profile_id=runtime_profile_id,
        log_fn=lambda message: print(f"[research_tick] {message}", flush=True),
    )
    report = judge_learner_sensitivity_ladder(bundle, **judge_policy)
    rubric, context_packet = _load_agentic_overlay(output_dir)
    report, judgment_brief = apply_agentic_judge(
        card=card,
        context_packet=context_packet,
        bundle=bundle,
        mechanical_report=report,
        evaluation_rubric=rubric,
    )

    progress_artifacts = write_progress_artifacts(
        output_dir,
        build_learner_ladder_progress_payload(
            experiment_id=card.experiment_id,
            metric_name=card.metric_name,
            regimes=list(regimes),
            current_step="judge_completed",
            completed_runs=[
                {
                    "regime_id": regime_id,
                    "config_name": next(
                        (regime.get("config_name", "") for regime in regimes if str(regime.get("regime_id")) == regime_id),
                        "",
                    ),
                    "metric_value": metric_value,
                    "status": "completed",
                }
                for regime_id, metric_value in bundle.summary.get("per_regime_metrics", {}).items()
            ],
            failures=list(bundle.summary.get("failures", [])),
            judge_decision=str(report.decision),
            human_review_required=bool(card.human_review_required),
        ),
    )
    bundle.metadata.update(
        {
            "task_plan_path": progress_artifacts["task_plan_path"],
            "progress_markdown_path": progress_artifacts["progress_markdown_path"],
            "handoff_path": progress_artifacts["handoff_path"],
        }
    )

    write_result_bundle(output_dir, bundle)
    write_judge_report_json(output_dir, report)
    write_judge_report_markdown(output_dir, report)
    write_judgment_artifacts(output_dir, judgment_brief)
    finished_at = utc_now_iso()
    duration_seconds = time.perf_counter() - started_perf
    manifest = build_run_manifest(
        card=card,
        card_path=card_path,
        repo_root=repo_root,
        output_dir=output_dir,
        judge_policy_path=judge_policy_path,
        invoked_command=" ".join(sys.argv) if sys.argv else "run_research_tick",
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_seconds=duration_seconds,
        metadata_overrides={
            "worker_runtime_profile_id": runtime_profile_id,
            "worker_python_bin": resolved_python_bin,
            "worker_preflight_path": str((output_dir / "preflight_report.json").resolve()),
        },
    )
    write_run_manifest(output_dir, manifest)
    _prime_agentic_artifacts(repo_root=repo_root, card=card, card_path=card_path)

    print(
        f"{card.experiment_id}: decision={report.decision} "
        f"completed={bundle.summary['completed_regime_count']} "
        f"best={bundle.summary['best_regime_id']} range={bundle.summary['regime_range']:.4f}"
    )
    return 0


def _run_feature_intervention_matrix(
    card_path: Path,
    *,
    python_bin_override: str = "",
    runtime_profile_id: str = "",
) -> int:
    repo_root = _discover_repo_root(card_path)
    card = load_experiment_card(card_path)
    input_path = resolve_repo_path(repo_root, card.input_path)
    output_dir = resolve_repo_path(repo_root, card.output_dir)
    judge_policy, judge_policy_path = _resolve_judge_policy(card, repo_root)
    metadata = dict(card.metadata)
    started_at = utc_now_iso()
    started_perf = time.perf_counter()

    _prime_agentic_artifacts(repo_root=repo_root, card=card, card_path=card_path)

    resolved_python_bin = str(python_bin_override or metadata.get("python_bin", sys.executable))
    requirements = build_worker_requirements(card, repo_root)
    preflight_report = probe_python_runtime(
        resolved_python_bin,
        required_modules=list(requirements["required_modules"]),
        require_cuda=bool(requirements["require_cuda"]),
    )
    preflight_report.update(
        {
            "runtime_profile_id": runtime_profile_id,
            "worker_script": requirements["worker_script"],
            "worker_script_path": requirements["worker_script_path"],
            "config_name": requirements["config_name"],
            "config_path": requirements["config_path"],
        }
    )
    write_json(output_dir / "preflight_report.json", preflight_report)
    if not preflight_report.get("passed", False):
        raise RuntimeError(f"worker runtime preflight failed: {preflight_report.get('reason', 'unknown reason')}")

    bundle = run_feature_intervention_matrix(
        experiment_id=card.experiment_id,
        subset_manifest_path=input_path,
        output_dir=output_dir,
        metric_name=card.metric_name,
        processed_data_root=str(resolve_repo_path(repo_root, str(metadata.get("processed_data_root", "data/data_feature")))),
        schema_path=str(resolve_repo_path(repo_root, str(metadata.get("schema_path", "docs/feature_schema/unified_processed_feature_schema.json")))),
        learner_variants=list(metadata.get("learner_variants", [])),
        probe_feature_axes=list(metadata.get("probe_feature_axes", [])),
        optional_probe_axes=list(metadata.get("optional_probe_axes", [])),
        tier_plan=dict(metadata.get("tier_plan", {})),
        python_bin=resolved_python_bin,
        worker_script=str(metadata.get("worker_script", "run_remix_training_experiment.py")),
        config_name=str(metadata.get("config_name", "feature_experiment_fast_cached_slide")),
        gpu_id=str(metadata.get("gpu_id", "0")),
        master_port_base=int(metadata.get("master_port_base", 29800)),
        result_name=str(metadata.get("result_name", "result.json")),
        runtime_profile_id=runtime_profile_id,
        num_classes=int(metadata.get("num_classes", 171)),
        log_fn=lambda message: print(f"[research_tick] {message}", flush=True),
    )
    report = judge_feature_intervention_matrix(bundle, **judge_policy)
    rubric, context_packet = _load_agentic_overlay(output_dir)
    report, judgment_brief = apply_agentic_judge(
        card=card,
        context_packet=context_packet,
        bundle=bundle,
        mechanical_report=report,
        evaluation_rubric=rubric,
    )

    trace_payload = {}
    execution_trace_path = str(bundle.metadata.get("execution_trace_path", ""))
    if execution_trace_path:
        trace_payload = load_json(execution_trace_path)
    progress_artifacts = write_progress_artifacts(
        output_dir,
        build_feature_intervention_progress_payload(
            experiment_id=card.experiment_id,
            metric_name=card.metric_name,
            learner_variants=list(metadata.get("learner_variants", [])),
            probe_axes=list(metadata.get("probe_feature_axes", [])),
            current_step="judge_completed",
            completed_runs=list(trace_payload.get("completed_runs", [])),
            failures=list(trace_payload.get("failures", [])),
            judge_decision=str(report.decision),
            human_review_required=bool(card.human_review_required),
        ),
    )
    bundle.metadata.update(
        {
            "task_plan_path": progress_artifacts["task_plan_path"],
            "progress_markdown_path": progress_artifacts["progress_markdown_path"],
            "handoff_path": progress_artifacts["handoff_path"],
        }
    )

    write_result_bundle(output_dir, bundle)
    write_judge_report_json(output_dir, report)
    write_judge_report_markdown(output_dir, report)
    write_judgment_artifacts(output_dir, judgment_brief)
    finished_at = utc_now_iso()
    duration_seconds = time.perf_counter() - started_perf
    manifest = build_run_manifest(
        card=card,
        card_path=card_path,
        repo_root=repo_root,
        output_dir=output_dir,
        judge_policy_path=judge_policy_path,
        invoked_command=" ".join(sys.argv) if sys.argv else "run_research_tick",
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_seconds=duration_seconds,
        metadata_overrides={
            "worker_runtime_profile_id": runtime_profile_id,
            "worker_python_bin": resolved_python_bin,
            "worker_preflight_path": str((output_dir / "preflight_report.json").resolve()),
        },
    )
    write_run_manifest(output_dir, manifest)
    _prime_agentic_artifacts(repo_root=repo_root, card=card, card_path=card_path)

    print(
        f"{card.experiment_id}: decision={report.decision} "
        f"tier={bundle.summary['tier_executed']} "
        f"cells_above_noise={bundle.summary['real_cells_above_noise_floor_count']} "
        f"best_ratio={bundle.summary['best_real_response_to_noise_ratio']:.4f}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    card_path = Path(args.experiment_card)
    card = load_experiment_card(card_path)

    if card.loop_kind == "noise_floor":
        return _run_noise_floor(card_path)
    if card.loop_kind == "same_subset_multi_seed":
        return _run_same_subset_multi_seed(
            card_path,
            python_bin_override=args.python_bin_override,
            runtime_profile_id=args.runtime_profile_id,
        )
    if card.loop_kind == "learner_sensitivity_ladder":
        return _run_learner_sensitivity_ladder(
            card_path,
            python_bin_override=args.python_bin_override,
            runtime_profile_id=args.runtime_profile_id,
        )
    if card.loop_kind == "feature_intervention_matrix":
        return _run_feature_intervention_matrix(
            card_path,
            python_bin_override=args.python_bin_override,
            runtime_profile_id=args.runtime_profile_id,
        )
    if card.loop_kind == "literature_radar":
        return _run_literature_radar(card_path)

    parser.error(f"unsupported loop_kind: {card.loop_kind}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
