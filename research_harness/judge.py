"""Fixed-rule judges for the thin research harness."""

from __future__ import annotations

from typing import Any, Dict, List

from .contracts import JudgeReport, ResultBundle


def judge_noise_floor(
    bundle: ResultBundle,
    *,
    minimum_labeled_runs: int = 30,
    narrow_std_threshold: float = 0.05,
    narrow_range_threshold: float = 0.20,
) -> JudgeReport:
    summary = dict(bundle.summary)
    count = int(summary.get("count", 0))
    stdev = float(summary.get("stdev", 0.0))
    observed_range = float(summary.get("range", 0.0))

    reasons: List[str] = []
    recommended_actions: List[str] = []

    if count < minimum_labeled_runs:
        reasons.append(
            f"Labeled run count {count} is below the minimum requirement {minimum_labeled_runs}."
        )
        recommended_actions.append("Collect more labeled runs before treating this as a stable floor.")
        return JudgeReport(
            experiment_id=bundle.experiment_id,
            decision="rerun",
            evidence_level="E1",
            result_summary=summary,
            reasons=reasons,
            recommended_actions=recommended_actions,
            requires_literature_radar=False,
        )

    evidence_level = "E2" if count < 100 else "E3"
    if stdev <= narrow_std_threshold and observed_range <= narrow_range_threshold:
        reasons.append(
            f"Observed std {stdev:.4f} and range {observed_range:.4f} are both below narrow-floor thresholds."
        )
        reasons.append(
            "The current metric spread is tight enough to justify moving to same-subset multi-seed and learner-sensitivity audits."
        )
        recommended_actions.extend(
            [
                "Promote EXP-P1-001 as completed and archive this summary as the current global-mIoU floor.",
                "Run same-subset multi-training-seed experiments to separate subset effect from training noise.",
                "Run learner sensitivity ladder before expanding downstream search or surrogate work.",
            ]
        )
        decision = "promote"
    else:
        reasons.append(
            f"Observed std {stdev:.4f} or range {observed_range:.4f} is wider than the current narrow-floor thresholds."
        )
        reasons.append(
            "The floor summary is still useful, but it should not yet be treated as a tight baseline."
        )
        recommended_actions.extend(
            [
                "Keep this summary as baseline context, but do not over-interpret small downstream deltas.",
                "Collect additional repeated runs or stratify the analysis before promoting downstream claims.",
            ]
        )
        decision = "park"

    return JudgeReport(
        experiment_id=bundle.experiment_id,
        decision=decision,
        evidence_level=evidence_level,
        result_summary=summary,
        reasons=reasons,
        recommended_actions=recommended_actions,
        requires_literature_radar=False,
    )


def judge_same_subset_multi_seed(
    bundle: ResultBundle,
    *,
    minimum_completed_runs: int = 3,
    global_floor_stdev_reference: float = 0.026,
    comparable_noise_ratio: float = 1.0,
) -> JudgeReport:
    summary = dict(bundle.summary)
    completed = int(summary.get("completed_seed_count", 0))
    stdev = float(summary.get("stdev", 0.0))
    reference = float(global_floor_stdev_reference)
    reasons: List[str] = []
    recommended_actions: List[str] = []

    if completed < minimum_completed_runs:
        reasons.append(
            f"Completed multi-seed runs {completed} are below the minimum requirement {minimum_completed_runs}."
        )
        recommended_actions.append("Finish the remaining seeds before drawing conclusions about training noise.")
        return JudgeReport(
            experiment_id=bundle.experiment_id,
            decision="rerun",
            evidence_level="E1",
            result_summary=summary,
            reasons=reasons,
            recommended_actions=recommended_actions,
        )

    evidence_level = "E2" if completed < 5 else "E3"
    ratio = float("inf") if reference <= 0 else stdev / reference
    summary["noise_to_global_floor_ratio"] = ratio

    if ratio >= float(comparable_noise_ratio):
        reasons.append(
            f"Multi-seed training noise stdev {stdev:.4f} is comparable to or larger than the global floor reference {reference:.4f}."
        )
        recommended_actions.extend(
            [
                "Treat small global-mIoU deltas cautiously; training noise may already explain much of the observed spread.",
                "Prioritize learner-sensitivity interventions before investing in downstream search complexity.",
            ]
        )
    else:
        reasons.append(
            f"Multi-seed training noise stdev {stdev:.4f} is meaningfully below the global floor reference {reference:.4f}."
        )
        recommended_actions.extend(
            [
                "Proceed to feature intervention and slice-leverage audits because data-composition effects may still be separable from training noise.",
                "Keep this summary as the training-noise baseline for future comparisons.",
            ]
        )

    return JudgeReport(
        experiment_id=bundle.experiment_id,
        decision="promote",
        evidence_level=evidence_level,
        result_summary=summary,
        reasons=reasons,
        recommended_actions=recommended_actions,
        requires_literature_radar=False,
    )


def judge_literature_radar(
    bundle: ResultBundle,
    *,
    minimum_ranked_results: int = 6,
    minimum_reproduce_candidates: int = 1,
    maximum_search_error_ratio: float = 0.5,
) -> JudgeReport:
    summary = dict(bundle.summary)
    query_count = int(summary.get("query_count", 0))
    ranked_result_count = int(summary.get("ranked_result_count", 0))
    reproduce_count = int(summary.get("reproduce_count", 0))
    search_error_ratio = float(summary.get("search_error_ratio", 0.0))

    reasons: List[str] = []
    recommended_actions: List[str] = []

    if query_count <= 0:
        reasons.append("Literature radar did not execute any queries.")
        recommended_actions.append("Rebuild the query plan before using this radar result.")
        return JudgeReport(
            experiment_id=bundle.experiment_id,
            decision="rerun",
            evidence_level="E1",
            result_summary=summary,
            reasons=reasons,
            recommended_actions=recommended_actions,
            requires_literature_radar=False,
        )

    if search_error_ratio > maximum_search_error_ratio:
        reasons.append(
            f"Search error ratio {search_error_ratio:.2f} is above the allowed threshold {maximum_search_error_ratio:.2f}."
        )
        recommended_actions.append("Repair literature retrieval or retry with a reduced query family before trusting the method ranking.")
        return JudgeReport(
            experiment_id=bundle.experiment_id,
            decision="rerun",
            evidence_level="E1",
            result_summary=summary,
            reasons=reasons,
            recommended_actions=recommended_actions,
            requires_literature_radar=False,
        )

    evidence_level = "E2" if ranked_result_count < 10 else "E3"
    if ranked_result_count >= minimum_ranked_results and reproduce_count >= minimum_reproduce_candidates:
        reasons.append(
            f"Literature radar ranked {ranked_result_count} candidate methods with {reproduce_count} reproduction-ready recommendations."
        )
        reasons.append("The method space is now broad enough to support focused read/reproduce decisions.")
        recommended_actions.extend(
            [
                "Open the top method cards and decide whether to admit one method into the isolated reproduction lane.",
                "Use the ranked list to inform debate on whether the current branch should keep, park, or pivot.",
            ]
        )
        decision = "promote"
    else:
        reasons.append(
            f"Literature radar produced {ranked_result_count} ranked results and {reproduce_count} reproduction-ready methods, which is below the current promotion floor."
        )
        recommended_actions.extend(
            [
                "Refine the query plan around the current failure mode before expanding the branch.",
                "Keep the current radar output as context, but do not treat it as sufficient space expansion yet.",
            ]
        )
        decision = "park"

    return JudgeReport(
        experiment_id=bundle.experiment_id,
        decision=decision,
        evidence_level=evidence_level,
        result_summary=summary,
        reasons=reasons,
        recommended_actions=recommended_actions,
        requires_literature_radar=False,
    )


def judge_learner_sensitivity_ladder(
    bundle: ResultBundle,
    *,
    minimum_completed_regimes: int = 3,
    training_noise_reference: float = 0.0089442719,
    meaningful_sensitivity_multiplier: float = 1.5,
) -> JudgeReport:
    summary = dict(bundle.summary)
    completed = int(summary.get("completed_regime_count", 0))
    regime_range = float(summary.get("regime_range", 0.0))
    threshold = float(training_noise_reference) * float(meaningful_sensitivity_multiplier)
    reasons: List[str] = []
    recommended_actions: List[str] = []

    if completed < minimum_completed_regimes:
        reasons.append(
            f"Completed regimes {completed} are below the minimum requirement {minimum_completed_regimes}."
        )
        recommended_actions.append("Finish the remaining ladder regimes before inferring learner sensitivity.")
        return JudgeReport(
            experiment_id=bundle.experiment_id,
            decision="rerun",
            evidence_level="E1",
            result_summary=summary,
            reasons=reasons,
            recommended_actions=recommended_actions,
        )

    evidence_level = "E2" if completed < 4 else "E3"
    summary["meaningful_sensitivity_threshold"] = threshold
    if regime_range > threshold:
        reasons.append(
            f"Observed learner-regime range {regime_range:.4f} is above the meaningful sensitivity threshold {threshold:.4f}."
        )
        reasons.append(
            f"Best regime `{summary.get('best_regime_id', '')}` outperformed baseline `{summary.get('baseline_regime_id', '')}` by {float(summary.get('best_minus_baseline', 0.0)):.4f}."
        )
        recommended_actions.extend(
            [
                "Promote the strongest learner regime as the next anchor for feature intervention experiments.",
                "Use this ladder result to decide whether protocol sensitivity or feature choice is the main bottleneck.",
            ]
        )
        decision = "promote"
    else:
        reasons.append(
            f"Observed learner-regime range {regime_range:.4f} is not meaningfully above the training-noise-derived threshold {threshold:.4f}."
        )
        recommended_actions.extend(
            [
                "Treat the current learner family as potentially too insensitive for strong composition claims.",
                "Consider triggering literature radar or alternative learners before expanding downstream search.",
            ]
        )
        decision = "park"

    return JudgeReport(
        experiment_id=bundle.experiment_id,
        decision=decision,
        evidence_level=evidence_level,
        result_summary=summary,
        reasons=reasons,
        recommended_actions=recommended_actions,
        requires_literature_radar=decision != "promote",
    )


def judge_feature_intervention_matrix(
    bundle: ResultBundle,
    *,
    minimum_completed_learner_variants: int = 3,
    minimum_probe_axes: int = 2,
    tier_a_requirements: Dict[str, Any] | None = None,
    design_mode: str = "",
    **_ignored: Any,
) -> JudgeReport:
    summary = dict(bundle.summary)
    tier_a_requirements = dict(tier_a_requirements or {})
    completed_variants = int(summary.get("completed_learner_variant_count", 0))
    completed_axes = int(summary.get("completed_probe_axis_count", 0))
    cells_above_floor = int(summary.get("real_cells_above_noise_floor_count", 0))
    best_ratio = float(summary.get("best_real_response_to_noise_ratio", 0.0))
    mean_off_target_drift_ratio = float(summary.get("mean_off_target_drift_ratio", 0.0))
    realized_delta_logged = bool(summary.get("realized_target_delta_logged_for_all_cells", False))

    minimum_ratio = float(tier_a_requirements.get("minimum_response_to_noise_ratio", 1.0))
    maximum_off_target_drift_ratio = float(tier_a_requirements.get("maximum_mean_off_target_drift_ratio", 1.0))
    require_realized_delta = bool(tier_a_requirements.get("require_realized_target_delta_logged", True))

    reasons: List[str] = []
    recommended_actions: List[str] = []

    if completed_variants < minimum_completed_learner_variants or completed_axes < minimum_probe_axes:
        reasons.append(
            f"Completed variants={completed_variants} or axes={completed_axes} are below the required floor ({minimum_completed_learner_variants} variants, {minimum_probe_axes} axes)."
        )
        recommended_actions.append("Finish the missing learner-axis cells before judging the Tier A screen.")
        return JudgeReport(
            experiment_id=bundle.experiment_id,
            decision="rerun",
            evidence_level="E1",
            result_summary=summary,
            reasons=reasons,
            recommended_actions=recommended_actions,
        )

    evidence_level = "E2"
    if require_realized_delta and not realized_delta_logged:
        reasons.append("Tier A screen did not log realized_target_delta for every learner-axis cell.")
        recommended_actions.append("Repair materialization logging before using this screen as evidence.")
        return JudgeReport(
            experiment_id=bundle.experiment_id,
            decision="rerun",
            evidence_level="E1",
            result_summary=summary,
            reasons=reasons,
            recommended_actions=recommended_actions,
        )

    if cells_above_floor > 0 and best_ratio >= minimum_ratio and mean_off_target_drift_ratio <= maximum_off_target_drift_ratio:
        reasons.append(
            f"Tier A found {cells_above_floor} learner-axis cells above the screen threshold with best response_to_noise_ratio={best_ratio:.4f}."
        )
        reasons.append(
            f"Mean off-target drift ratio {mean_off_target_drift_ratio:.4f} stays within the Tier A caution ceiling {maximum_off_target_drift_ratio:.4f}."
        )
        recommended_actions.extend(
            [
                "Treat this as a successful Tier A screen and promote the strongest learner-axis cells to Tier B confirmation.",
                "Keep teacher frozen and validation full while adding shuffled and matched-random controls.",
            ]
        )
        decision = "promote"
    else:
        reasons.append(
            f"Tier A did not surface a strong enough signal: best response_to_noise_ratio={best_ratio:.4f}, cells_above_floor={cells_above_floor}, mean_off_target_drift_ratio={mean_off_target_drift_ratio:.4f}."
        )
        recommended_actions.extend(
            [
                "Park this branch at Tier A and reconsider learner adaptability, probe-axis choice, or materialization fidelity.",
                "Do not escalate to Tier B until at least one learner-axis cell clearly exceeds its learner-specific noise floor.",
            ]
        )
        decision = "park"

    return JudgeReport(
        experiment_id=bundle.experiment_id,
        decision=decision,
        evidence_level=evidence_level,
        result_summary=summary,
        reasons=reasons,
        recommended_actions=recommended_actions,
        requires_literature_radar=False,
    )
