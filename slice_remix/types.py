from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RemixContext:
    budget: int
    sample_ids: list[str]
    slice_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateAction:
    baseline_mixture: list[float]
    target_mixture: list[float]
    delta_q: list[float]
    donors: list[int] = field(default_factory=list)
    receivers: list[int] = field(default_factory=list)
    amplitude: float = 0.0
    support_size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseRow:
    baseline_mixture: list[float]
    target_mixture: list[float]
    delta_q: list[float]
    portrait_shift: dict[str, list[float]]
    measured_gain: float
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecommendationResult:
    candidate_id: str
    baseline_mixture: list[float]
    target_mixture: list[float]
    delta_q: list[float]
    predicted_gain_mean: float
    predicted_gain_std: float
    risk_adjusted_score: float
    context: dict[str, Any] = field(default_factory=dict)
    portrait_summary: dict[str, Any] = field(default_factory=dict)
    rationale: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    ranked_candidates: list[dict[str, Any]] = field(default_factory=list)
    search_tree: dict[str, Any] = field(default_factory=dict)
