from __future__ import annotations

import json
from typing import Any

from .metrics import extract_metric_value
from .surrogate import cross_validate_surrogate, describe_surrogate_algorithm


def _baseline_seed(row: dict[str, Any]) -> str:
    context = row.get("context", {})
    if isinstance(context, dict) and "baseline_seed" in context:
        return str(context["baseline_seed"])
    return "unknown"


def _load_metric_value(result_path: str, metric_path: str) -> float:
    with open(result_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return extract_metric_value(payload, metric_path)


def _build_actual_rank_map(rows: list[dict[str, Any]]) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: float(row["measured_gain"]), reverse=True)
    return {str(row["candidate_id"]): rank for rank, row in enumerate(ordered, start=1)}


def summarize_recommendation_actuals(
    *,
    recommendation: dict[str, Any],
    response_rows: list[dict[str, Any]],
    baseline_result_path: str | None = None,
    recommended_result_path: str | None = None,
    metric_path: str | None = None,
) -> dict[str, Any]:
    candidate_id = str(recommendation.get("candidate_id", ""))
    baseline_seed = str(recommendation.get("context", {}).get("baseline_seed", "unknown"))
    rows_by_candidate = {str(row["candidate_id"]): row for row in response_rows}
    observed_row = rows_by_candidate.get(candidate_id)

    actual_comparison = {
        "source": "unavailable",
        "baseline_metric_value": None,
        "candidate_metric_value": None,
        "actual_gain": None,
        "prediction_error": None,
        "absolute_prediction_error": None,
        "metric_path": metric_path,
    }

    if observed_row is not None:
        actual_gain = float(observed_row["measured_gain"])
        prediction_error = float(recommendation["predicted_gain_mean"]) - actual_gain
        actual_comparison.update(
            {
                "source": "response_dataset",
                "baseline_metric_value": observed_row.get("baseline_metric_value"),
                "candidate_metric_value": observed_row.get("candidate_metric_value"),
                "actual_gain": actual_gain,
                "prediction_error": prediction_error,
                "absolute_prediction_error": abs(prediction_error),
                "metric_path": observed_row.get("metric_path", metric_path),
            }
        )
        return actual_comparison

    if baseline_result_path and recommended_result_path and metric_path:
        baseline_metric_value = _load_metric_value(baseline_result_path, metric_path)
        candidate_metric_value = _load_metric_value(recommended_result_path, metric_path)
        actual_gain = candidate_metric_value - baseline_metric_value
        prediction_error = float(recommendation["predicted_gain_mean"]) - actual_gain
        actual_comparison.update(
            {
                "source": "validation_results",
                "baseline_metric_value": baseline_metric_value,
                "candidate_metric_value": candidate_metric_value,
                "actual_gain": actual_gain,
                "prediction_error": prediction_error,
                "absolute_prediction_error": abs(prediction_error),
                "metric_path": metric_path,
                "baseline_result_path": baseline_result_path,
                "candidate_result_path": recommended_result_path,
            }
        )
        return actual_comparison

    return actual_comparison


def build_ranked_candidate_report(
    *,
    recommendation: dict[str, Any],
    response_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_seed = str(recommendation.get("context", {}).get("baseline_seed", "unknown"))
    baseline_rows = [row for row in response_rows if _baseline_seed(row) == baseline_seed]
    rank_map = _build_actual_rank_map(baseline_rows) if baseline_rows else {}
    rows_by_candidate = {str(row["candidate_id"]): row for row in baseline_rows}

    ranked_report: list[dict[str, Any]] = []
    for predicted_rank, candidate in enumerate(recommendation.get("ranked_candidates", []), start=1):
        candidate_id = str(candidate.get("candidate_id", ""))
        observed_row = rows_by_candidate.get(candidate_id)
        entry = dict(candidate)
        entry["predicted_rank"] = predicted_rank
        if observed_row is None:
            entry["observed_actual_gain"] = None
            entry["prediction_error"] = None
            entry["actual_rank_within_baseline"] = None
            entry["observed_baseline_metric_value"] = None
            entry["observed_candidate_metric_value"] = None
        else:
            actual_gain = float(observed_row["measured_gain"])
            entry["observed_actual_gain"] = actual_gain
            entry["prediction_error"] = float(candidate.get("predicted_gain_mean", 0.0)) - actual_gain
            entry["actual_rank_within_baseline"] = rank_map.get(candidate_id)
            entry["observed_baseline_metric_value"] = observed_row.get("baseline_metric_value")
            entry["observed_candidate_metric_value"] = observed_row.get("candidate_metric_value")
        ranked_report.append(entry)
    return ranked_report


def build_analysis_report(
    *,
    response_rows: list[dict[str, Any]],
    recommendation: dict[str, Any],
    model_name: str,
    bootstrap_models: int,
    kappa: float,
    top_k: int,
    baseline_result_path: str | None = None,
    recommended_result_path: str | None = None,
    metric_path: str | None = None,
) -> dict[str, Any]:
    labeled_rows = [row for row in response_rows if row.get("measured_gain") is not None]
    actual_comparison = summarize_recommendation_actuals(
        recommendation=recommendation,
        response_rows=labeled_rows,
        baseline_result_path=baseline_result_path,
        recommended_result_path=recommended_result_path,
        metric_path=metric_path,
    )
    ranked_candidates = build_ranked_candidate_report(
        recommendation=recommendation,
        response_rows=labeled_rows,
    )

    return {
        "surrogate": {
            "algorithm": describe_surrogate_algorithm(model_name, bootstrap_models=bootstrap_models),
            "training_row_count": len(labeled_rows),
            "model_artifact_path": recommendation.get("context", {}).get("surrogate_output_path"),
            "cross_validation": cross_validate_surrogate(
                labeled_rows,
                model_name=model_name,
                bootstrap_models=bootstrap_models,
                group_key="baseline_seed",
                top_k=top_k,
                kappa=kappa,
            ),
        },
        "recommendation": {
            "candidate_id": recommendation.get("candidate_id"),
            "baseline_seed": recommendation.get("context", {}).get("baseline_seed"),
            "budget": recommendation.get("context", {}).get("budget"),
            "predicted_gain_mean": recommendation.get("predicted_gain_mean"),
            "predicted_gain_std": recommendation.get("predicted_gain_std"),
            "risk_adjusted_score": recommendation.get("risk_adjusted_score"),
            "delta_q": recommendation.get("delta_q"),
            "rationale": recommendation.get("rationale", {}),
            "portrait_summary": recommendation.get("portrait_summary", {}),
            "actual_comparison": actual_comparison,
        },
        "ranked_candidates": ranked_candidates,
    }
