"""Contracts for the thin, agent-centered research harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class ExperimentCard:
    experiment_id: str
    name: str
    phase: str
    owner: str
    loop_kind: str
    status: str = "queued"
    hypothesis: str = ""
    budget_tier: str = "Tier A"
    input_path: str = ""
    metric_name: str = "mIoU"
    output_dir: str = ""
    judge_policy_path: str = ""
    requires_debate: bool = False
    debate_bundle_path: str = ""
    human_review_required: bool = False
    phase_completion_candidate: bool = False
    priority: int = 100
    depends_on: List[str] = field(default_factory=list)
    next_eligible_at_utc: str = ""
    attempt_count: int = 0
    max_attempts: int = 0
    last_attempt_id: str = ""
    status_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Legacy fallback only. New cards should point to a locked judge policy file.
    judge_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentCard":
        return cls(
            experiment_id=str(payload["experiment_id"]),
            name=str(payload["name"]),
            phase=str(payload["phase"]),
            owner=str(payload["owner"]),
            loop_kind=str(payload["loop_kind"]),
            status=str(payload.get("status", "queued")),
            hypothesis=str(payload.get("hypothesis", "")),
            budget_tier=str(payload.get("budget_tier", "Tier A")),
            input_path=str(payload.get("input_path", "")),
            metric_name=str(payload.get("metric_name", "mIoU")),
            output_dir=str(payload.get("output_dir", "")),
            judge_policy_path=str(payload.get("judge_policy_path", "")),
            requires_debate=bool(payload.get("requires_debate", False)),
            debate_bundle_path=str(payload.get("debate_bundle_path", "")),
            human_review_required=bool(payload.get("human_review_required", False)),
            phase_completion_candidate=bool(payload.get("phase_completion_candidate", False)),
            priority=int(payload.get("priority", 100)),
            depends_on=[str(item) for item in payload.get("depends_on", [])],
            next_eligible_at_utc=str(payload.get("next_eligible_at_utc", "")),
            attempt_count=int(payload.get("attempt_count", 0)),
            max_attempts=int(payload.get("max_attempts", 0)),
            last_attempt_id=str(payload.get("last_attempt_id", "")),
            status_history=list(payload.get("status_history", [])),
            metadata=dict(payload.get("metadata", {})),
            judge_config=dict(payload.get("judge_config", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResultBundle:
    experiment_id: str
    loop_kind: str
    input_path: str
    metric_name: str
    summary: Dict[str, Any]
    sample_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeReport:
    experiment_id: str
    decision: str
    evidence_level: str
    result_summary: Dict[str, Any]
    reasons: List[str]
    recommended_actions: List[str]
    requires_literature_radar: bool = False
    protocol_contamination: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifest:
    experiment_id: str
    loop_kind: str
    card_path: str
    output_dir: str
    judge_policy_path: str
    invoked_command: str
    repo_root: str
    git_sha: str
    git_branch: str
    git_is_dirty: bool
    python_version: str
    platform: str
    hostname: str
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttemptManifest:
    attempt_id: str
    experiment_id: str
    session_id: str
    card_path: str
    output_dir: str
    attempt_dir: str
    runtime_profile_id: str
    python_bin: str
    status: str
    started_at_utc: str
    finished_at_utc: str = ""
    reason: str = ""
    exit_code: int = 0
    paths: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProposalRecord:
    proposal_id: str
    proposal_class: str
    status: str
    phase: str
    title: str
    rationale: str
    source_experiment_ids: List[str] = field(default_factory=list)
    target_experiment_id: str = ""
    target_card_path: str = ""
    auto_materialized: bool = False
    created_at_utc: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
