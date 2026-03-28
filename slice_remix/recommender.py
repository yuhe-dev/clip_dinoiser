from __future__ import annotations

from typing import Any

from .types import RecommendationResult


def score_candidate(
    *,
    predicted_gain_mean: float,
    predicted_gain_std: float,
    l1_shift: float,
    support_size: int,
    kappa: float,
    lambda_l1: float,
    lambda_support: float,
) -> float:
    return (
        float(predicted_gain_mean)
        - float(kappa) * float(predicted_gain_std)
        - float(lambda_l1) * float(l1_shift)
        - float(lambda_support) * float(support_size)
    )


def rank_candidates(
    candidates: list[dict[str, Any]],
    surrogate: Any,
    *,
    kappa: float = 0.0,
    lambda_l1: float = 0.0,
    lambda_support: float = 0.0,
) -> list[dict[str, Any]]:
    mean_preds = surrogate.predict_mean(candidates)
    std_preds = surrogate.predict_std(candidates)

    ranked: list[dict[str, Any]] = []
    for candidate, mean_pred, std_pred in zip(candidates, mean_preds, std_preds):
        scored = dict(candidate)
        scored["predicted_gain_mean"] = float(mean_pred)
        scored["predicted_gain_std"] = float(std_pred)
        scored["risk_adjusted_score"] = score_candidate(
            predicted_gain_mean=float(mean_pred),
            predicted_gain_std=float(std_pred),
            l1_shift=float(candidate.get("l1_shift", 0.0)),
            support_size=int(candidate.get("support_size", 0)),
            kappa=kappa,
            lambda_l1=lambda_l1,
            lambda_support=lambda_support,
        )
        ranked.append(scored)

    ranked.sort(key=lambda row: row["risk_adjusted_score"], reverse=True)
    return ranked


def build_recommendation_result(candidate: dict[str, Any]) -> RecommendationResult:
    return RecommendationResult(
        candidate_id=str(candidate.get("candidate_id", "")),
        baseline_mixture=list(candidate.get("baseline_mixture", [])),
        target_mixture=list(candidate.get("target_mixture", [])),
        delta_q=list(candidate.get("delta_q", [])),
        predicted_gain_mean=float(candidate.get("predicted_gain_mean", 0.0)),
        predicted_gain_std=float(candidate.get("predicted_gain_std", 0.0)),
        risk_adjusted_score=float(candidate.get("risk_adjusted_score", 0.0)),
        context=dict(candidate.get("context", {})),
        portrait_summary=dict(candidate.get("portrait_summary", {})),
        rationale=dict(candidate.get("rationale", {})),
        execution=dict(candidate.get("execution", {})),
        ranked_candidates=list(candidate.get("ranked_candidates", [])),
        search_tree=dict(candidate.get("search_tree", {})),
    )
