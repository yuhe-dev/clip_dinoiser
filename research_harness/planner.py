"""Structured agentic planning artifacts for research cards."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .agentic_registry import summarize_key_value_pairs, write_agentic_artifact, write_agentic_markdown
from .contracts import ExperimentCard
from .registry import load_json, resolve_repo_path
from .runtime import utc_now_iso


def _load_optional_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


def _context_summary(context_packet: Dict[str, Any]) -> List[str]:
    facts = list(context_packet.get("task_snapshot", {}).get("recent_facts", []))
    blockers = list(context_packet.get("task_snapshot", {}).get("blockers", []))
    merged: List[str] = []
    merged.extend([str(item) for item in facts[:5]])
    merged.extend([f"blocker:{item}" for item in blockers[:3]])
    return merged


def _design_mode(card: ExperimentCard) -> str:
    return str(card.metadata.get("design_mode", "")).strip()


def _design_spec_path(card: ExperimentCard, context_packet: Dict[str, Any]) -> str:
    return str(card.metadata.get("design_spec_path", "") or context_packet.get("design_spec_path", ""))


def _variant_ids(card: ExperimentCard) -> List[str]:
    variants = card.metadata.get("learner_variants", [])
    return [str(item.get("variant_id", "")) for item in variants if str(item.get("variant_id", ""))]


def _axis_ids(card: ExperimentCard, key: str = "probe_feature_axes") -> List[str]:
    axes = card.metadata.get(key, [])
    return [str(item.get("axis_id", "")) for item in axes if str(item.get("axis_id", ""))]


def _control_families(card: ExperimentCard) -> List[str]:
    return [str(item) for item in card.metadata.get("control_families", []) if str(item)]


def _loop_recipe(card: ExperimentCard) -> Dict[str, Any]:
    if _design_mode(card) == "minimal_learner_adaptability_audit":
        return {
            "objective": (
                "Run a minimal learner adaptability audit that compares whether more adaptable learner variants "
                "convert matched probe-axis feature interventions into stronger training response than the current head-only learner."
            ),
            "variables_changed": [
                "learner trainable scope",
                "feature-guided subset composition along probe axes",
            ],
            "variables_frozen": [
                "phase",
                "benchmark contract",
                "validation_mode=full",
                "teacher_policy=freeze_found_and_dino",
                "subset_budget=1000",
                "human review stop",
            ],
            "execution_recipe": [
                "estimate learner-specific noise floor for each learner variant",
                "materialize high/low subset pairs for each probe axis",
                "run Tier A with real_feature_guided only and feature_pair_seed=[0]",
                "advance learner-axis cells to Tier B only if real response exceeds that learner's noise floor",
                "run Tier B with real, shuffled, and matched-random controls across feature_pair_seeds=[0,1,2]",
                "run Tier C only for at most two promoted learner-axis cells across feature_pair_seeds=[0,1,2,3,4]",
            ],
            "confound_guardrails": [
                "do not mix algorithm mutation with protocol mutation in one step",
                "do not change teacher targets in this audit",
                "do not compare learners using a shared noise floor",
                "do not interpret intended feature shift without logging realized_target_delta and off_target_drift",
            ],
        }
    if card.loop_kind == "noise_floor":
        return {
            "objective": "Summarize the current labeled random-subset floor.",
            "variables_changed": ["none; analysis-only baseline consolidation"],
            "variables_frozen": ["training protocol", "benchmark split", "metric definition"],
            "execution_recipe": [
                "load labeled random-subset rows",
                "extract metric values",
                "compute global floor summary",
                "judge whether spread is narrow enough for downstream audits",
            ],
            "confound_guardrails": [
                "require stable labeled result source",
                "do not mix benchmark versions",
            ],
        }
    if card.loop_kind == "same_subset_multi_seed":
        return {
            "objective": "Estimate training noise while holding subset composition fixed.",
            "variables_changed": ["training_seed"],
            "variables_frozen": ["subset manifest", "config", "metric", "budget", "runtime profile"],
            "execution_recipe": [
                "materialize per-seed manifests from a fixed subset",
                "launch one training run per seed",
                "reuse completed runs when provenance matches",
                "summarize seed-wise metric spread",
            ],
            "confound_guardrails": [
                "fixed subset manifest across all seeds",
                "matched runtime profile and config",
                "completion sentinel must match provenance",
            ],
        }
    if card.loop_kind == "learner_sensitivity_ladder":
        return {
            "objective": "Estimate whether learner/protocol choices alter response sensitivity more than known training noise.",
            "variables_changed": ["training regime", "config_name", "effective training budget"],
            "variables_frozen": ["subset manifest", "metric", "validation split", "runtime family"],
            "execution_recipe": [
                "reuse one fixed subset manifest as the anchor",
                "run the same subset under several learner/protocol regimes",
                "compare regime range against the known training-noise baseline",
                "promote only if the ladder shows meaningful sensitivity",
            ],
            "confound_guardrails": [
                "do not change subset composition while auditing learner sensitivity",
                "keep runtime family and metric constant across regimes",
            ],
        }
    if card.loop_kind == "literature_radar":
        return {
            "objective": "Expand the method space when the current branch is bottlenecked.",
            "variables_changed": ["query plan", "venue priority", "method ranking"],
            "variables_frozen": ["current phase", "current bottleneck definition"],
            "execution_recipe": [
                "build a query plan from stage, failure mode, and modality",
                "search external literature sources",
                "rank candidate methods",
                "emit method cards and reproduction recommendations",
            ],
            "confound_guardrails": [
                "separate infrastructure failures from scientific bottlenecks",
                "isolate reproduction into design-only lane until admitted",
            ],
        }
    return {
        "objective": "Design a new research branch under the current phase contract.",
        "variables_changed": ["to be specified"],
        "variables_frozen": ["phase", "benchmark contract", "human review stop"],
        "execution_recipe": [
            "refine hypothesis",
            "define mutation scope",
            "freeze evaluation rubric",
            "decide whether the branch is design-only or executable",
        ],
        "confound_guardrails": [
            "do not mix algorithm mutation with protocol mutation in one step",
        ],
    }


def build_hypothesis_brief(
    card: ExperimentCard,
    *,
    context_packet: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    context_packet = context_packet or {}
    if _design_mode(card) == "minimal_learner_adaptability_audit":
        return {
            "generated_by": "agentic_planner_v1",
            "generated_at_utc": utc_now_iso(),
            "experiment_id": card.experiment_id,
            "phase": card.phase,
            "loop_kind": card.loop_kind,
            "status": card.status,
            "problem_statement": (
                "Under Phase 1, determine whether the current head-only learner is too rigid to convert "
                "feature-defined data composition differences into stable training response, and whether more "
                "adaptable learner variants reveal stronger response to real probe-axis interventions than to "
                "shuffled or matched-random controls."
            ),
            "hypothesis": card.hypothesis or "hypothesis pending refinement",
            "trigger_context": _context_summary(context_packet),
            "success_signal": (
                "At least one learner variant exceeds its own noise floor on at least one real probe axis, and in "
                "confirm stage the real intervention achieves response_to_noise_ratio >= 2.0 while beating shuffled "
                "and matched-random controls."
            ),
            "failure_signal": (
                "All learner variants remain below their own noise floors, or real features fail to outperform "
                "shuffled/random controls, or off-target drift remains too large to support causal interpretation."
            ),
            "open_questions": [
                "Which learner adaptability level first shows composition response above noise floor?",
                "Which probe axis shows earlier stable signal: quality_sharpness or difficulty_small_object?",
                "Do real features outperform shuffled and matched-random controls?",
                "Is the bottleneck learner rigidity, feature invalidity, or both?",
            ],
            "source_paths": {
                "context_packet": context_packet.get("source_path", ""),
                "card_path": context_packet.get("card_path", ""),
                "design_spec_path": _design_spec_path(card, context_packet),
            },
        }
    return {
        "generated_by": "agentic_planner_v1",
        "generated_at_utc": utc_now_iso(),
        "experiment_id": card.experiment_id,
        "phase": card.phase,
        "loop_kind": card.loop_kind,
        "status": card.status,
        "problem_statement": (
            f"Under {card.phase}, determine whether {card.loop_kind} can reduce uncertainty in the current research object."
        ),
        "hypothesis": card.hypothesis or "hypothesis pending refinement",
        "trigger_context": _context_summary(context_packet),
        "success_signal": "Produce evidence strong enough to update keep/park/kill for the current branch.",
        "failure_signal": "Results remain below noise floor or fail to distinguish scientific signal from infrastructure noise.",
        "open_questions": [
            "What variable is being changed in this loop?",
            "What comparison is frozen?",
            "What evidence would justify promoting the branch?",
        ],
        "source_paths": {
            "context_packet": context_packet.get("source_path", ""),
            "card_path": context_packet.get("card_path", ""),
        },
    }


def build_design_pack(
    card: ExperimentCard,
    *,
    context_packet: Dict[str, Any] | None = None,
    hypothesis_brief: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    context_packet = context_packet or {}
    hypothesis_brief = hypothesis_brief or build_hypothesis_brief(card, context_packet=context_packet)
    recipe = _loop_recipe(card)
    payload = {
        "generated_by": "agentic_planner_v1",
        "generated_at_utc": utc_now_iso(),
        "experiment_id": card.experiment_id,
        "phase": card.phase,
        "loop_kind": card.loop_kind,
        "design_class": "design_only" if bool(card.metadata.get("design_only", False)) else "executable_candidate",
        "objective": recipe["objective"],
        "mutation_scope": {
            "changed": list(recipe["variables_changed"]),
            "frozen": list(recipe["variables_frozen"]),
        },
        "execution_recipe": list(recipe["execution_recipe"]),
        "confound_guardrails": list(recipe["confound_guardrails"]),
        "expected_signal": {
            "primary": hypothesis_brief.get("success_signal", ""),
            "secondary": "Produce structured artifacts that can be judged and debated.",
        },
        "runtime_requirements": {
            "input_path": card.input_path,
            "output_dir": card.output_dir,
            "judge_policy_path": card.judge_policy_path,
        },
        "source_paths": {
            "hypothesis_brief": "",
            "context_packet": context_packet.get("source_path", ""),
        },
    }
    if _design_mode(card) == "minimal_learner_adaptability_audit":
        payload["learner_variants"] = list(card.metadata.get("learner_variants", []))
        payload["probe_axes"] = list(card.metadata.get("probe_feature_axes", []))
        optional_axes = list(card.metadata.get("optional_probe_axes", []))
        if optional_axes:
            payload["optional_probe_axes"] = optional_axes
        payload["control_families"] = _control_families(card)
        payload["reporting_metrics"] = [str(item) for item in card.metadata.get("reporting_metrics", [])]
        payload["tier_plan"] = dict(card.metadata.get("tier_plan", {}))
        payload["source_paths"]["design_spec_path"] = _design_spec_path(card, context_packet)
    return payload


def build_evaluation_rubric(
    card: ExperimentCard,
    *,
    repo_root: str | Path,
    context_packet: Dict[str, Any] | None = None,
    design_pack: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    context_packet = context_packet or {}
    design_pack = design_pack or build_design_pack(card, context_packet=context_packet)
    judge_policy = {}
    if card.judge_policy_path:
        judge_policy = _load_optional_json(resolve_repo_path(repo_root, card.judge_policy_path))
    criteria: Dict[str, List[str]] = {
        "success_criteria": [],
        "caution_criteria": [],
        "failure_criteria": [],
    }
    judge_contract: Dict[str, Any] = {
        "contract_type": "generic_design_review",
        "thresholds": {},
    }
    if card.loop_kind == "noise_floor":
        criteria["success_criteria"].append("Have at least the minimum labeled run count required by the judge policy.")
        criteria["success_criteria"].append("Observed spread is narrow enough to justify downstream audits.")
        criteria["caution_criteria"].append("If spread is wide, baseline is still useful but should not anchor small-Delta claims.")
        criteria["failure_criteria"].append("Insufficient labeled runs to establish a stable floor.")
        judge_contract = {
            "contract_type": "noise_floor",
            "thresholds": {
                "minimum_labeled_runs": int(judge_policy.get("minimum_labeled_runs", 30)),
                "narrow_std_threshold": float(judge_policy.get("narrow_std_threshold", 0.05)),
                "narrow_range_threshold": float(judge_policy.get("narrow_range_threshold", 0.20)),
            },
        }
    elif card.loop_kind == "same_subset_multi_seed":
        criteria["success_criteria"].append("Completed multi-seed count meets the minimum requirement.")
        criteria["success_criteria"].append("Training noise is meaningfully below the global floor reference.")
        criteria["caution_criteria"].append("If noise is comparable to the floor, downstream claims need stronger controls.")
        criteria["failure_criteria"].append("Seed runs are incomplete or corrupted.")
        judge_contract = {
            "contract_type": "same_subset_multi_seed",
            "thresholds": {
                "minimum_completed_runs": int(judge_policy.get("minimum_completed_runs", 3)),
                "global_floor_stdev_reference": float(judge_policy.get("global_floor_stdev_reference", 0.026)),
                "comparable_noise_ratio": float(judge_policy.get("comparable_noise_ratio", 1.0)),
            },
        }
    elif card.loop_kind == "literature_radar":
        criteria["success_criteria"].append("At least several methods are retrieved, ranked, and scoped to the current bottleneck.")
        criteria["success_criteria"].append("At least one method card is recommended for reproduction or deeper study.")
        criteria["caution_criteria"].append("Avoid over-expanding into unrelated adjacent research areas.")
        criteria["failure_criteria"].append("Search only returns low-relevance or duplicate methods.")
        judge_contract = {
            "contract_type": "literature_radar",
            "thresholds": {
                "minimum_ranked_results": int(judge_policy.get("minimum_ranked_results", 6)),
                "minimum_reproduce_candidates": int(judge_policy.get("minimum_reproduce_candidates", 1)),
                "maximum_search_error_ratio": float(judge_policy.get("maximum_search_error_ratio", 0.5)),
            },
        }
    elif card.loop_kind == "learner_sensitivity_ladder":
        criteria["success_criteria"].append("Completed regimes meet the minimum ladder size.")
        criteria["success_criteria"].append("Regime-to-regime range exceeds the known training-noise baseline by a meaningful margin.")
        criteria["caution_criteria"].append("If range stays near the training-noise floor, current learner/protocol may be too insensitive.")
        criteria["failure_criteria"].append("Regime executions are incomplete or corrupted.")
        judge_contract = {
            "contract_type": "learner_sensitivity_ladder",
            "thresholds": {
                "minimum_completed_regimes": int(judge_policy.get("minimum_completed_regimes", 3)),
                "training_noise_reference": float(judge_policy.get("training_noise_reference", 0.0089442719)),
                "meaningful_sensitivity_multiplier": float(judge_policy.get("meaningful_sensitivity_multiplier", 1.5)),
            },
        }
    elif _design_mode(card) == "minimal_learner_adaptability_audit":
        criteria["success_criteria"].append(
            "At least one learner-axis cell passes Tier B with response_to_noise_ratio >= 2.0, directional_consistency >= 0.67, and real_feature_guided beating both shuffled and matched-random controls."
        )
        criteria["success_criteria"].append("Teacher policy remains frozen and validation mode remains full.")
        criteria["caution_criteria"].append("Tier A is screening evidence only and cannot by itself justify a strong research claim.")
        criteria["caution_criteria"].append("If off_target_drift_ratio > 1.0, treat the result as weak evidence only.")
        criteria["caution_criteria"].append("If learner response increases together with learner noise, prefer a more controllable learner over a simply stronger one.")
        criteria["failure_criteria"].append("All learner-axis cells remain below their own noise floors.")
        criteria["failure_criteria"].append("Real features fail to beat shuffled and matched-random controls.")
        criteria["failure_criteria"].append("Realized intervention fidelity remains too poor to support causal interpretation.")
        judge_contract = {
            "contract_type": "learner_adaptability_audit",
            "thresholds": {
                "minimum_real_axes_with_signal": int(
                    ((judge_policy.get("promote_requirements") or {}).get("minimum_real_axes_with_signal", 1))
                ),
                "tier_a.minimum_response_to_noise_ratio": float(
                    ((judge_policy.get("tier_a_requirements") or {}).get("minimum_response_to_noise_ratio", 1.0))
                ),
                "tier_b.minimum_response_to_noise_ratio": float(
                    ((judge_policy.get("promote_requirements") or {}).get("minimum_response_to_noise_ratio", 2.0))
                ),
                "tier_b.minimum_directional_consistency": float(
                    ((judge_policy.get("promote_requirements") or {}).get("minimum_directional_consistency", 0.67))
                ),
                "tier_b.require_real_beats_shuffled": bool(
                    ((judge_policy.get("promote_requirements") or {}).get("require_real_beats_shuffled", True))
                ),
                "tier_b.require_real_beats_random": bool(
                    ((judge_policy.get("promote_requirements") or {}).get("require_real_beats_random", True))
                ),
                "maximum_mean_off_target_drift_ratio": float(
                    ((judge_policy.get("caution_requirements") or {}).get("maximum_mean_off_target_drift_ratio", 1.0))
                ),
            },
        }
    else:
        criteria["success_criteria"].append("Design clarifies the changed variable, frozen comparison, and expected signal.")
        criteria["caution_criteria"].append("Do not promote without a task-specific evidence floor.")
        criteria["failure_criteria"].append("Design remains ambiguous or unexecutable.")

    payload = {
        "generated_by": "agentic_planner_v1",
        "generated_at_utc": utc_now_iso(),
        "experiment_id": card.experiment_id,
        "phase": card.phase,
        "loop_kind": card.loop_kind,
        "primary_metric": card.metric_name,
        "comparison_contract": {
            "phase": card.phase,
            "depends_on": list(card.depends_on),
            "design_class": design_pack.get("design_class", ""),
        },
        "judge_contract": judge_contract,
        "judge_policy_snapshot": judge_policy,
        "success_criteria": criteria["success_criteria"],
        "caution_criteria": criteria["caution_criteria"],
        "failure_criteria": criteria["failure_criteria"],
        "promote_rule": "Promote only when success criteria are met and no major protocol contamination is detected.",
        "park_rule": "Park when interesting signal exists but the current phase or evidence depth is insufficient.",
        "kill_rule": "Kill when repeated execution still fails to produce interpretable signal under a valid protocol.",
        "source_paths": {
            "context_packet": context_packet.get("source_path", ""),
            "judge_policy_path": card.judge_policy_path,
        },
    }
    if _design_mode(card) == "minimal_learner_adaptability_audit":
        payload["metric_definitions"] = {
            "signed_response": "mIoU(high) - mIoU(low)",
            "composition_response_amplitude": "abs(signed_response)",
            "response_to_noise_ratio": "composition_response_amplitude / learner_specific_noise_std",
            "directional_consistency": "fraction of seeds whose signed_response matches the majority sign",
            "feature_validity_advantage": "real_feature_guided.response_to_noise_ratio - max(shuffled_feature_guided.response_to_noise_ratio, matched_random_control.response_to_noise_ratio)",
        }
        payload["promote_rule"] = (
            "Promote the current learner-feature coupling branch only when at least one learner-axis cell satisfies "
            "the Tier B success criteria under a valid protocol."
        )
        payload["park_rule"] = (
            "Park when there is weak signal but only Tier A evidence or insufficient fidelity/seed depth to justify a strong claim."
        )
        payload["kill_rule"] = (
            "Kill the current probe axis or learner branch when repeated valid execution still leaves real features "
            "below both controls and the learner-specific noise floor."
        )
        payload["source_paths"]["design_spec_path"] = _design_spec_path(card, context_packet)
    return payload


def render_agentic_markdown(
    *,
    title: str,
    payload: Dict[str, Any],
    key_order: List[str],
) -> str:
    lines = [f"# {title}", ""]
    summary = summarize_key_value_pairs(payload, key_order)
    if summary:
        lines.append(summary.rstrip())
        lines.append("")
    for key, value in payload.items():
        if key in key_order:
            continue
        if value in (None, "", [], {}):
            continue
        lines.append(f"## {key}")
        lines.append("")
        if isinstance(value, list):
            lines.extend([f"- {item}" for item in value])
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                lines.append(f"- `{sub_key}`: {sub_value}")
        else:
            lines.append(str(value))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_planning_artifacts(
    card: ExperimentCard,
    *,
    repo_root: str | Path,
    context_packet: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    output_dir = resolve_repo_path(repo_root, card.output_dir) if card.output_dir else Path(repo_root)
    context_packet = dict(context_packet or {})
    hypothesis_brief = build_hypothesis_brief(card, context_packet=context_packet)
    design_pack = build_design_pack(card, context_packet=context_packet, hypothesis_brief=hypothesis_brief)
    rubric = build_evaluation_rubric(
        card,
        repo_root=repo_root,
        context_packet=context_packet,
        design_pack=design_pack,
    )
    hypothesis_path = write_agentic_artifact(output_dir, "hypothesis_brief", hypothesis_brief)
    design_path = write_agentic_artifact(output_dir, "design_pack", design_pack)
    rubric_path = write_agentic_artifact(output_dir, "evaluation_rubric", rubric)
    write_agentic_markdown(
        output_dir,
        "hypothesis_brief",
        render_agentic_markdown(
            title="Hypothesis Brief",
            payload=hypothesis_brief,
            key_order=["experiment_id", "phase", "loop_kind", "status", "hypothesis"],
        ),
    )
    write_agentic_markdown(
        output_dir,
        "design_pack",
        render_agentic_markdown(
            title="Design Pack",
            payload=design_pack,
            key_order=["experiment_id", "phase", "loop_kind", "design_class", "objective"],
        ),
    )
    write_agentic_markdown(
        output_dir,
        "evaluation_rubric",
        render_agentic_markdown(
            title="Evaluation Rubric",
            payload=rubric,
            key_order=["experiment_id", "phase", "loop_kind", "primary_metric"],
        ),
    )
    return {
        "hypothesis_brief_path": str(hypothesis_path.resolve()),
        "design_pack_path": str(design_path.resolve()),
        "evaluation_rubric_path": str(rubric_path.resolve()),
    }
