"""Task-level progress artifacts for the research harness conductor layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .runtime import utc_now_iso

def _write_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return target


def _status_marker(status: str) -> str:
    mapping = {
        "completed": "[x]",
        "running": "[>]",
        "failed": "[!]",
        "blocked": "[!]",
        "reused": "[x]",
        "pending": "[ ]",
    }
    return mapping.get(status, "[ ]")


def _normalize_task_status(raw_status: str) -> str:
    text = str(raw_status).strip().lower()
    if text in {"completed", "done"}:
        return "completed"
    if text in {"reused_existing_result", "reused"}:
        return "reused"
    if text in {"running", "in_progress"}:
        return "running"
    if text in {"failed", "blocked"}:
        return "failed"
    return "pending"


def write_progress_artifacts(output_dir: str | Path, payload: Dict[str, Any]) -> Dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    enriched = dict(payload)
    enriched.setdefault("updated_at_utc", utc_now_iso())
    enriched.setdefault("research_state", "execution")
    enriched.setdefault("next_state", "")
    enriched.setdefault("state_history", [])
    task_plan_path = _write_json(root / "task_plan.json", enriched)
    markdown_path = root / "progress.md"
    markdown_path.write_text(render_progress_markdown(enriched), encoding="utf-8")
    handoff_path = root / "handoff.md"
    handoff_path.write_text(render_handoff_markdown(enriched), encoding="utf-8")
    return {
        "task_plan_path": str(task_plan_path.resolve()),
        "progress_markdown_path": str(markdown_path.resolve()),
        "handoff_path": str(handoff_path.resolve()),
    }


def render_progress_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Experiment Progress",
        "",
        f"- `experiment_id`: {payload.get('experiment_id', '')}",
        f"- `loop_kind`: {payload.get('loop_kind', '')}",
        f"- `current_step`: {payload.get('current_step', '')}",
        f"- `research_state`: {payload.get('research_state', '')}",
        f"- `next_state`: {payload.get('next_state', '')}",
        f"- `updated_at_utc`: {payload.get('updated_at_utc', '')}",
    ]
    if payload.get("next_action"):
        lines.append(f"- `next_action`: {payload.get('next_action', '')}")
    if payload.get("acceptance_status"):
        lines.append(f"- `acceptance_status`: {payload.get('acceptance_status', '')}")
    lines.extend(["", "## Tasks", ""])

    for task in payload.get("tasks", []):
        status = _normalize_task_status(str(task.get("status", "")))
        marker = _status_marker(status)
        title = str(task.get("title", task.get("task_id", "")))
        stage = str(task.get("stage", "")).strip()
        stage_label = f"[{stage}]" if stage else ""
        lines.append(f"- {marker} `{task.get('task_id', '')}` {stage_label} {title}".strip())
        detail = str(task.get("detail", "")).strip()
        if detail:
            lines.append(f"  - {detail}")

    blockers = list(payload.get("blockers", []))
    if blockers:
        lines.extend(["", "## Blockers", ""])
        for blocker in blockers:
            lines.append(f"- {blocker}")

    facts = list(payload.get("recent_facts", []))
    if facts:
        lines.extend(["", "## Recent Facts", ""])
        for fact in facts:
            lines.append(f"- {fact}")

    return "\n".join(lines).rstrip() + "\n"


def render_handoff_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Task Handoff",
        "",
        f"- `experiment_id`: {payload.get('experiment_id', '')}",
        f"- `loop_kind`: {payload.get('loop_kind', '')}",
        f"- `current_step`: {payload.get('current_step', '')}",
        f"- `research_state`: {payload.get('research_state', '')}",
        f"- `next_action`: {payload.get('next_action', '')}",
        f"- `updated_at_utc`: {payload.get('updated_at_utc', '')}",
    ]
    blockers = list(payload.get("blockers", []))
    if blockers:
        lines.extend(["", "## Blockers", ""])
        for blocker in blockers:
            lines.append(f"- {blocker}")
    facts = list(payload.get("recent_facts", []))
    if facts:
        lines.extend(["", "## Recent Facts", ""])
        for fact in facts:
            lines.append(f"- {fact}")
    return "\n".join(lines).rstrip() + "\n"


def build_noise_floor_progress_payload(
    *,
    experiment_id: str,
    metric_name: str,
    summary: Dict[str, Any],
    judge_decision: str = "",
    human_review_required: bool = False,
) -> Dict[str, Any]:
    tasks = [
        {
            "task_id": "load_rows",
            "title": "Load historical runs",
            "status": "completed",
            "stage": "audit",
            "detail": f"loaded {int(summary.get('count', 0))} labeled rows",
        },
        {
            "task_id": "summarize_floor",
            "title": "Summarize noise floor",
            "status": "completed",
            "stage": "verification",
            "detail": (
                f"{metric_name} mean={summary.get('mean', 0.0):.4f} "
                f"std={summary.get('stdev', 0.0):.4f} range={summary.get('range', 0.0):.4f}"
            ),
        },
        {
            "task_id": "judge_results",
            "title": "Judge evidence level",
            "status": "completed" if judge_decision else "pending",
            "stage": "judgment",
            "detail": f"decision={judge_decision or 'pending'}",
        },
        {
            "task_id": "human_acceptance",
            "title": "Human review stop",
            "status": "pending" if human_review_required else "completed",
            "stage": "acceptance",
            "detail": "await human validation" if human_review_required else "not required",
        },
    ]
    research_state = "judgment" if judge_decision else "verification"
    if human_review_required:
        research_state = "acceptance"
    return {
        "experiment_id": experiment_id,
        "loop_kind": "noise_floor",
        "current_step": "judge_completed" if judge_decision else "summarized",
        "next_action": "wait_human_review" if human_review_required else "done",
        "acceptance_status": "awaiting_human_review" if human_review_required else "not_required",
        "research_state": research_state,
        "next_state": "acceptance" if human_review_required else "",
        "tasks": tasks,
        "recent_facts": [
            f"{metric_name} count={int(summary.get('count', 0))}",
            f"{metric_name} mean={summary.get('mean', 0.0):.4f}",
            f"{metric_name} std={summary.get('stdev', 0.0):.4f}",
            f"{metric_name} range={summary.get('range', 0.0):.4f}",
        ],
        "blockers": [],
    }


def build_same_subset_progress_payload(
    *,
    experiment_id: str,
    metric_name: str,
    training_seeds: Iterable[int],
    current_step: str,
    completed_runs: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    judge_decision: str = "",
    human_review_required: bool = False,
) -> Dict[str, Any]:
    completed_by_seed = {int(item["training_seed"]): dict(item) for item in completed_runs}
    failed_by_seed = {int(item["training_seed"]): dict(item) for item in failures}
    launch_prefix = "launch_seed_"
    running_seed = None
    if str(current_step).startswith(launch_prefix):
        try:
            running_seed = int(str(current_step).split(launch_prefix, 1)[1])
        except ValueError:
            running_seed = None

    tasks: List[Dict[str, Any]] = []
    for seed in [int(item) for item in training_seeds]:
        if seed in completed_by_seed:
            item = completed_by_seed[seed]
            tasks.append(
                {
                    "task_id": f"train_seed_{seed:02d}",
                    "title": f"Train fixed subset with seed {seed}",
                    "status": _normalize_task_status(str(item.get("status", "completed"))),
                    "stage": "execution",
                    "detail": f"{metric_name}={float(item.get('metric_value', 0.0)):.4f}",
                }
            )
            continue
        if seed in failed_by_seed:
            item = failed_by_seed[seed]
            tasks.append(
                {
                    "task_id": f"train_seed_{seed:02d}",
                    "title": f"Train fixed subset with seed {seed}",
                    "status": "failed",
                    "stage": "execution",
                    "detail": f"exit_code={int(item.get('exit_code', 0))}",
                }
            )
            continue
        tasks.append(
            {
                "task_id": f"train_seed_{seed:02d}",
                "title": f"Train fixed subset with seed {seed}",
                "status": "running" if running_seed == seed else "pending",
                "stage": "execution",
                "detail": "launched under current runtime profile" if running_seed == seed else "",
            }
        )

    execution_complete = len(completed_runs) + len(failures) >= len([int(seed) for seed in training_seeds])
    tasks.extend(
        [
            {
                "task_id": "summarize_results",
                "title": "Summarize multi-seed response",
                "status": "completed" if execution_complete else "pending",
                "stage": "verification",
                "detail": (
                    f"completed={len(completed_runs)} failed={len(failures)}"
                    if execution_complete
                    else f"completed={len(completed_runs)} failed={len(failures)}"
                ),
            },
            {
                "task_id": "judge_results",
                "title": "Judge training-noise audit",
                "status": "completed" if judge_decision else "pending",
                "stage": "judgment",
                "detail": f"decision={judge_decision or 'pending'}",
            },
            {
                "task_id": "human_acceptance",
                "title": "Human review stop",
                "status": "pending" if human_review_required else "completed",
                "stage": "acceptance",
                "detail": "await human validation" if human_review_required else "not required",
            },
        ]
    )

    recent_facts = [
        f"completed_seed_count={len(completed_runs)}",
        f"failure_count={len(failures)}",
    ]
    if completed_runs:
        last = completed_runs[-1]
        recent_facts.append(
            f"last_completed_seed={int(last.get('training_seed', -1))} "
            f"{metric_name}={float(last.get('metric_value', 0.0)):.4f}"
        )

    blockers = [
        f"training_seed={int(item.get('training_seed', -1))} failed with exit_code={int(item.get('exit_code', 0))}"
        for item in failures
    ]

    next_action = "wait_human_review" if judge_decision and human_review_required else (
        "judge_results" if execution_complete and not judge_decision else "continue_seed_execution"
    )
    research_state = "execution"
    if execution_complete:
        research_state = "verification"
    if judge_decision:
        research_state = "judgment"
    if human_review_required and judge_decision:
        research_state = "acceptance"

    return {
        "experiment_id": experiment_id,
        "loop_kind": "same_subset_multi_seed",
        "current_step": current_step,
        "next_action": next_action,
        "acceptance_status": "awaiting_human_review" if human_review_required and judge_decision else "not_ready",
        "research_state": research_state,
        "next_state": "acceptance" if human_review_required and judge_decision else "",
        "tasks": tasks,
        "recent_facts": recent_facts,
        "blockers": blockers,
    }


def build_literature_progress_payload(
    *,
    experiment_id: str,
    current_step: str,
    query_count: int,
    ranked_result_count: int,
    reproduce_count: int,
    search_error_count: int,
    judge_decision: str = "",
    human_review_required: bool = False,
) -> Dict[str, Any]:
    search_complete = current_step in {"search_completed", "judge_completed"}
    tasks = [
        {
            "task_id": "plan_queries",
            "title": "Plan literature queries from the active bottleneck",
            "status": "completed",
            "stage": "design",
            "detail": f"query_count={int(query_count)}",
        },
        {
            "task_id": "retrieve_papers",
            "title": "Retrieve and rank candidate methods",
            "status": "completed" if search_complete else "running",
            "stage": "execution",
            "detail": (
                f"ranked_results={int(ranked_result_count)} reproduce={int(reproduce_count)} errors={int(search_error_count)}"
            ),
        },
        {
            "task_id": "judge_radar",
            "title": "Judge literature radar output",
            "status": "completed" if judge_decision else "pending",
            "stage": "judgment",
            "detail": f"decision={judge_decision or 'pending'}",
        },
        {
            "task_id": "human_acceptance",
            "title": "Human review stop",
            "status": "pending" if human_review_required and judge_decision else "completed",
            "stage": "acceptance",
            "detail": "await human validation" if human_review_required and judge_decision else "not required",
        },
    ]

    research_state = "execution"
    if search_complete:
        research_state = "verification"
    if judge_decision:
        research_state = "judgment"
    if human_review_required and judge_decision:
        research_state = "acceptance"

    blockers: List[str] = []
    if search_error_count:
        blockers.append(f"literature_search_errors={int(search_error_count)}")

    return {
        "experiment_id": experiment_id,
        "loop_kind": "literature_radar",
        "current_step": current_step,
        "next_action": "wait_human_review" if human_review_required and judge_decision else ("review_method_cards" if judge_decision else "execute_query_plan"),
        "acceptance_status": "awaiting_human_review" if human_review_required and judge_decision else "not_required",
        "research_state": research_state,
        "next_state": "acceptance" if human_review_required and judge_decision else "",
        "tasks": tasks,
        "recent_facts": [
            f"query_count={int(query_count)}",
            f"ranked_result_count={int(ranked_result_count)}",
            f"reproduce_count={int(reproduce_count)}",
            f"search_error_count={int(search_error_count)}",
        ],
        "blockers": blockers,
    }


def build_learner_ladder_progress_payload(
    *,
    experiment_id: str,
    metric_name: str,
    regimes: List[Dict[str, Any]],
    current_step: str,
    completed_runs: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    judge_decision: str = "",
    human_review_required: bool = False,
) -> Dict[str, Any]:
    completed_by_regime = {str(item["regime_id"]): dict(item) for item in completed_runs}
    failed_by_regime = {str(item["regime_id"]): dict(item) for item in failures}
    tasks: List[Dict[str, Any]] = []
    for regime in regimes:
        regime_id = str(regime["regime_id"])
        title = f"Run learner regime {regime_id}"
        if regime_id in completed_by_regime:
            item = completed_by_regime[regime_id]
            tasks.append(
                {
                    "task_id": f"regime_{regime_id}",
                    "title": title,
                    "status": _normalize_task_status(str(item.get("status", "completed"))),
                    "stage": "execution",
                    "detail": (
                        f"config={item.get('config_name', '')} "
                        f"{metric_name}={float(item.get('metric_value', 0.0)):.4f}"
                    ),
                }
            )
            continue
        if regime_id in failed_by_regime:
            item = failed_by_regime[regime_id]
            tasks.append(
                {
                    "task_id": f"regime_{regime_id}",
                    "title": title,
                    "status": "failed",
                    "stage": "execution",
                    "detail": f"config={item.get('config_name', '')} exit_code={int(item.get('exit_code', 0))}",
                }
            )
            continue
        is_running = current_step == f"launch_regime_{regime_id}"
        tasks.append(
            {
                "task_id": f"regime_{regime_id}",
                "title": title,
                "status": "running" if is_running else "pending",
                "stage": "execution",
                "detail": f"config={regime.get('config_name', '')}",
            }
        )

    execution_complete = len(completed_runs) + len(failures) >= len(regimes)
    tasks.extend(
        [
            {
                "task_id": "summarize_results",
                "title": "Summarize learner-sensitivity ladder",
                "status": "completed" if execution_complete else "pending",
                "stage": "verification",
                "detail": f"completed={len(completed_runs)} failed={len(failures)}",
            },
            {
                "task_id": "judge_results",
                "title": "Judge learner-sensitivity signal",
                "status": "completed" if judge_decision else "pending",
                "stage": "judgment",
                "detail": f"decision={judge_decision or 'pending'}",
            },
            {
                "task_id": "human_acceptance",
                "title": "Human review stop",
                "status": "pending" if human_review_required and judge_decision else "completed",
                "stage": "acceptance",
                "detail": "await human validation" if human_review_required and judge_decision else "not required",
            },
        ]
    )
    research_state = "execution"
    if execution_complete:
        research_state = "verification"
    if judge_decision:
        research_state = "judgment"
    if human_review_required and judge_decision:
        research_state = "acceptance"
    return {
        "experiment_id": experiment_id,
        "loop_kind": "learner_sensitivity_ladder",
        "current_step": current_step,
        "next_action": "wait_human_review" if human_review_required and judge_decision else ("judge_results" if execution_complete and not judge_decision else "continue_regime_execution"),
        "acceptance_status": "awaiting_human_review" if human_review_required and judge_decision else "not_required",
        "research_state": research_state,
        "next_state": "acceptance" if human_review_required and judge_decision else "",
        "tasks": tasks,
        "recent_facts": [
            f"completed_regime_count={len(completed_runs)}",
            f"failure_count={len(failures)}",
        ],
        "blockers": [
            f"regime_id={item.get('regime_id', '')} failed with exit_code={int(item.get('exit_code', 0))}"
            for item in failures
        ],
    }


def build_feature_intervention_progress_payload(
    *,
    experiment_id: str,
    metric_name: str,
    learner_variants: List[Dict[str, Any]],
    probe_axes: List[Dict[str, Any]],
    current_step: str,
    completed_runs: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    judge_decision: str = "",
    human_review_required: bool = False,
) -> Dict[str, Any]:
    tasks: List[Dict[str, Any]] = []
    completed_noise = {
        str(item.get("learner_variant_id", "")): dict(item)
        for item in completed_runs
        if str(item.get("stage", "")) == "noise_floor"
    }
    for variant in learner_variants:
        variant_id = str(variant.get("variant_id", ""))
        tasks.append(
            {
                "task_id": f"noise_{variant_id}",
                "title": f"Estimate learner noise for {variant_id}",
                "status": "completed" if variant_id in completed_noise else ("running" if current_step.startswith("noise_") else "pending"),
                "stage": "execution",
                "detail": (
                    f"{metric_name}={float(completed_noise[variant_id].get('metric_value', 0.0)):.4f}"
                    if variant_id in completed_noise
                    else f"modules={','.join(str(item) for item in variant.get('trainable_modules', []))}"
                ),
            }
        )

    completed_pairs = {
        (
            str(item.get("learner_variant_id", "")),
            str(item.get("axis_id", "")),
            str(item.get("direction", "")),
        ): dict(item)
        for item in completed_runs
        if str(item.get("stage", "")) == "feature_pair"
    }
    for variant in learner_variants:
        variant_id = str(variant.get("variant_id", ""))
        for axis in probe_axes:
            axis_id = str(axis.get("axis_id", ""))
            for direction in ("high", "low"):
                key = (variant_id, axis_id, direction)
                tasks.append(
                    {
                        "task_id": f"pair_{variant_id}_{axis_id}_{direction}",
                        "title": f"Run {variant_id} on {axis_id} ({direction})",
                        "status": "completed" if key in completed_pairs else ("running" if current_step.endswith(f"{axis_id}_{direction}") else "pending"),
                        "stage": "execution",
                        "detail": (
                            f"{metric_name}={float(completed_pairs[key].get('metric_value', 0.0)):.4f}"
                            if key in completed_pairs
                            else "real_feature_guided"
                        ),
                    }
                )

    execution_complete = not failures and len(completed_pairs) >= (len(learner_variants) * len(probe_axes) * 2)
    tasks.extend(
        [
            {
                "task_id": "summarize_results",
                "title": "Summarize learner adaptability screen",
                "status": "completed" if execution_complete else "pending",
                "stage": "verification",
                "detail": f"completed_runs={len(completed_runs)} failures={len(failures)}",
            },
            {
                "task_id": "judge_results",
                "title": "Judge Tier A feature intervention signal",
                "status": "completed" if judge_decision else "pending",
                "stage": "judgment",
                "detail": f"decision={judge_decision or 'pending'}",
            },
            {
                "task_id": "human_acceptance",
                "title": "Human review stop",
                "status": "pending" if human_review_required and judge_decision else "completed",
                "stage": "acceptance",
                "detail": "await human validation" if human_review_required and judge_decision else "not required",
            },
        ]
    )

    research_state = "execution"
    if execution_complete:
        research_state = "verification"
    if judge_decision:
        research_state = "judgment"
    if human_review_required and judge_decision:
        research_state = "acceptance"
    return {
        "experiment_id": experiment_id,
        "loop_kind": "feature_intervention_matrix",
        "current_step": current_step,
        "next_action": (
            "wait_human_review"
            if human_review_required and judge_decision
            else ("judge_results" if execution_complete and not judge_decision else "continue_feature_intervention_execution")
        ),
        "acceptance_status": "awaiting_human_review" if human_review_required and judge_decision else "not_required",
        "research_state": research_state,
        "next_state": "acceptance" if human_review_required and judge_decision else "",
        "tasks": tasks,
        "recent_facts": [
            f"completed_run_count={len(completed_runs)}",
            f"failure_count={len(failures)}",
            f"learner_variant_count={len(learner_variants)}",
            f"probe_axis_count={len(probe_axes)}",
        ],
        "blockers": [
            f"run_id={item.get('run_id', '')} failed"
            for item in failures
        ],
    }
