"""Phase-locked proposal generation for the autonomous research harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from .contracts import ExperimentCard, ProposalRecord
from .registry import load_experiment_card, load_json, write_experiment_card, write_json
from .runtime import utc_now_iso


def load_proposal_policy(path: str | Path) -> Dict[str, Any]:
    payload = load_json(path)
    payload.setdefault("rules", [])
    payload.setdefault("dynamic_rules", {})
    return payload


def _existing_card_map(scan_dir: str | Path) -> Dict[str, Tuple[Path, ExperimentCard]]:
    root = Path(scan_dir)
    if not root.exists():
        return {}
    result: Dict[str, Tuple[Path, ExperimentCard]] = {}
    for candidate in sorted(root.glob("*.json")):
        card = load_experiment_card(candidate)
        result[card.experiment_id] = (candidate, card)
    return result


def _runtime_entry_map(runtime_index: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(entry["experiment_id"]): dict(entry) for entry in runtime_index.get("entries", [])}


def _rule_is_triggered(
    rule: Dict[str, Any],
    *,
    runtime_by_id: Dict[str, Dict[str, Any]],
    cards_by_id: Dict[str, Tuple[Path, ExperimentCard]],
) -> tuple[bool, str]:
    target_experiment_id = str(rule.get("target_experiment_id", ""))
    if not target_experiment_id:
        return False, "missing_target_experiment_id"
    if target_experiment_id in cards_by_id:
        return False, "target_already_exists"

    for source_id in [str(item) for item in rule.get("trigger_on_completed", [])]:
        entry = runtime_by_id.get(source_id)
        if not entry:
            return False, f"missing_source:{source_id}"
        if str(entry.get("status", "")) != "completed":
            return False, f"source_not_completed:{source_id}:{entry.get('status', '')}"
        required_decisions = [str(item) for item in rule.get("required_decisions", [])]
        if required_decisions and str(entry.get("judge_decision", "")) not in required_decisions:
            return False, f"source_decision_blocked:{source_id}:{entry.get('judge_decision', '')}"
    return True, "triggered"


def _build_card_from_rule(rule: Dict[str, Any]) -> ExperimentCard:
    experiment_id = str(rule["target_experiment_id"])
    loop_kind = str(rule["loop_kind"])
    output_dir = str(rule.get("output_dir", "")).strip()
    if not output_dir:
        output_dir = f"artifacts/research_harness/{experiment_id}_{loop_kind}"
    debate_bundle_path = str(rule.get("debate_bundle_path", "")).strip()
    if bool(rule.get("requires_debate", True)) and not debate_bundle_path:
        debate_bundle_path = f".slicetune/debates/{experiment_id}_bundle.json"
    return ExperimentCard(
        experiment_id=experiment_id,
        name=str(rule["name"]),
        phase=str(rule["phase"]),
        owner=str(rule.get("owner", "Auto Proposer")),
        loop_kind=loop_kind,
        status=str(rule.get("target_status", "planned")),
        hypothesis=str(rule.get("hypothesis", "")),
        budget_tier=str(rule.get("budget_tier", "Tier B")),
        input_path=str(rule.get("input_path", "")),
        metric_name=str(rule.get("metric_name", "mIoU")),
        output_dir=output_dir,
        judge_policy_path=str(rule.get("judge_policy_path", "")),
        requires_debate=bool(rule.get("requires_debate", True)),
        debate_bundle_path=debate_bundle_path,
        human_review_required=bool(rule.get("human_review_required", False)),
        phase_completion_candidate=bool(rule.get("phase_completion_candidate", False)),
        priority=int(rule.get("priority", 100)),
        depends_on=[str(item) for item in rule.get("depends_on", [])],
        metadata=dict(rule.get("metadata", {})),
    )


def _proposal_repo_root(scan_dir: str | Path) -> Path:
    scan_dir = Path(scan_dir).resolve()
    if scan_dir.name == "experiments" and scan_dir.parent.name == ".slicetune":
        return scan_dir.parent.parent
    return scan_dir.parent


def build_proposals(
    *,
    runtime_index: Dict[str, Any],
    proposal_policy: Dict[str, Any],
    scan_dir: str | Path,
    task_board: Dict[str, Any] | None = None,
) -> List[ProposalRecord]:
    cards_by_id = _existing_card_map(scan_dir)
    runtime_by_id = _runtime_entry_map(runtime_index)
    proposals: List[ProposalRecord] = []
    for rule in proposal_policy.get("rules", []):
        triggered, reason = _rule_is_triggered(
            rule,
            runtime_by_id=runtime_by_id,
            cards_by_id=cards_by_id,
        )
        if not triggered:
            continue
        proposal_id = str(rule.get("proposal_id") or f"PRO-{rule['target_experiment_id']}")
        proposals.append(
            ProposalRecord(
                proposal_id=proposal_id,
                proposal_class=str(rule.get("proposal_class", "draft_only")),
                status="proposed",
                phase=str(rule.get("phase", "")),
                title=str(rule.get("name", proposal_id)),
                rationale=str(rule.get("rationale", reason)),
                source_experiment_ids=[str(item) for item in rule.get("trigger_on_completed", [])],
                target_experiment_id=str(rule.get("target_experiment_id", "")),
                created_at_utc=utc_now_iso(),
                metadata={
                    "rule": dict(rule),
                    "trigger_reason": reason,
                },
            )
        )
    proposals.extend(
        _build_dynamic_proposals(
            runtime_index=runtime_index,
            proposal_policy=proposal_policy,
            scan_dir=scan_dir,
            task_board=task_board or {},
            cards_by_id=cards_by_id,
        )
    )
    return proposals


def _build_dynamic_proposals(
    *,
    runtime_index: Dict[str, Any],
    proposal_policy: Dict[str, Any],
    scan_dir: str | Path,
    task_board: Dict[str, Any],
    cards_by_id: Dict[str, Tuple[Path, ExperimentCard]],
) -> List[ProposalRecord]:
    proposals: List[ProposalRecord] = []
    dynamic_rules = dict(proposal_policy.get("dynamic_rules", {}))
    radar_rule = dict(dynamic_rules.get("literature_radar", {}))
    if not radar_rule.get("enabled", False):
        return proposals

    task_board_map = {
        str(entry.get("experiment_id", "")): dict(entry)
        for entry in task_board.get("entries", [])
        if entry.get("experiment_id")
    }
    trigger_decisions = {str(item) for item in radar_rule.get("trigger_decisions", [])}
    trigger_status_prefixes = tuple(str(item) for item in radar_rule.get("trigger_status_prefixes", []))
    retry_exhaustion_statuses = {str(item) for item in radar_rule.get("retry_exhaustion_statuses", ["blocked_retry_limit"])}
    min_attempt_count = int(radar_rule.get("min_attempt_count", 0))
    proposal_class = str(radar_rule.get("proposal_class", "literature_radar"))
    phase_fallback = str(radar_rule.get("phase", ""))

    for entry in runtime_index.get("entries", []):
        experiment_id = str(entry.get("experiment_id", ""))
        if not experiment_id or experiment_id not in cards_by_id:
            continue
        proposal_id = f"LIT-{experiment_id}"
        if any(proposal.proposal_id == proposal_id for proposal in proposals):
            continue
        proposal_path = Path(scan_dir).parent / "proposals" / f"{proposal_id}.json"
        if proposal_path.exists():
            continue

        status = str(entry.get("status", ""))
        decision = str(entry.get("judge_decision", ""))
        attempt_count = int(entry.get("attempt_count", 0))
        task_entry = task_board_map.get(experiment_id, {})

        triggered = False
        reasons: List[str] = []
        if decision and decision in trigger_decisions:
            triggered = True
            reasons.append(f"judge_decision={decision}")
        if trigger_status_prefixes and any(status.startswith(prefix) for prefix in trigger_status_prefixes):
            triggered = True
            reasons.append(f"status={status}")
        if min_attempt_count > 0 and attempt_count >= min_attempt_count and status in retry_exhaustion_statuses:
            triggered = True
            reasons.append(f"attempt_count={attempt_count}")
        if not triggered:
            continue

        phase = str(entry.get("phase", "")) or phase_fallback
        target_experiment_id = f"EXP-LIT-{experiment_id}"
        title = f"Trigger literature radar for {experiment_id}"
        rationale = (
            "Automatic literature radar proposal after repeated failures or non-promotable outcome: "
            + ", ".join(reasons)
        )
        if task_entry.get("research_state"):
            rationale += f"; research_state={task_entry.get('research_state')}"
        proposals.append(
            ProposalRecord(
                proposal_id=proposal_id,
                proposal_class=proposal_class,
                status="proposed",
                phase=phase,
                title=title,
                rationale=rationale,
                source_experiment_ids=[experiment_id],
                target_experiment_id=target_experiment_id,
                created_at_utc=utc_now_iso(),
                metadata={
                    "trigger_reasons": reasons,
                    "source_status": status,
                    "source_decision": decision,
                    "generated_card": {
                        "experiment_id": target_experiment_id,
                        "name": title,
                        "phase": phase,
                        "owner": "Literature Radar",
                        "loop_kind": "literature_radar",
                        "status": "planned",
                        "budget_tier": "Tier A",
                        "output_dir": f"artifacts/research_harness/{target_experiment_id}_literature_radar",
                        "judge_policy_path": "",
                        "requires_debate": True,
                        "debate_bundle_path": f".slicetune/debates/{target_experiment_id}_bundle.json",
                        "human_review_required": True,
                        "phase_completion_candidate": False,
                        "priority": 20,
                        "depends_on": [experiment_id],
                        "metadata": {
                            "design_only": False,
                            "proposal_origin": "dynamic_literature_radar",
                            "source_experiment_id": experiment_id,
                            "trigger_reasons": reasons,
                            "search_backend": "openalex",
                            "max_results_per_query": 8,
                        },
                    },
                },
            )
        )
    return proposals


def materialize_proposals(
    proposals: List[ProposalRecord],
    *,
    proposal_policy: Dict[str, Any],
    scan_dir: str | Path,
    proposals_dir: str | Path,
) -> List[ProposalRecord]:
    root = Path(scan_dir)
    root.mkdir(parents=True, exist_ok=True)
    proposal_root = Path(proposals_dir)
    proposal_root.mkdir(parents=True, exist_ok=True)
    repo_root = _proposal_repo_root(scan_dir)

    for proposal in proposals:
        rule = dict(proposal.metadata.get("rule", {}))
        if proposal.target_experiment_id:
            target_path = root / f"{proposal.target_experiment_id}.json"
            proposal.target_card_path = str(target_path.resolve())
            if not target_path.exists():
                if proposal.proposal_class == "literature_radar" and proposal.metadata.get("generated_card"):
                    card = ExperimentCard.from_dict(dict(proposal.metadata.get("generated_card", {})))
                else:
                    card = _build_card_from_rule(rule)
                write_experiment_card(target_path, card)
                if card.requires_debate and card.debate_bundle_path:
                    debate_path = Path(card.debate_bundle_path)
                    if not debate_path.is_absolute():
                        debate_path = repo_root / card.debate_bundle_path
                    if not debate_path.exists():
                        write_json(
                            debate_path,
                            {
                                "decision": "pending",
                                "round_count": 0,
                                "reviewer_count": 0,
                                "artifact_paths": {
                                    "design_card": "",
                                    "review_cards": [],
                                    "arbiter_decision": "",
                                },
                                "created_by": "auto_proposer",
                                "created_at_utc": utc_now_iso(),
                            },
                        )
                proposal.auto_materialized = True
                proposal.status = "materialized"
        write_json(proposal_root / f"{proposal.proposal_id}.json", proposal.to_dict())
    return proposals


def build_proposal_index(proposals: List[ProposalRecord]) -> Dict[str, Any]:
    return {
        "generated_at_utc": utc_now_iso(),
        "proposal_count": len(proposals),
        "proposals": [proposal.to_dict() for proposal in proposals],
    }


def write_proposal_index(path: str | Path, payload: Dict[str, Any]) -> Path:
    write_json(path, payload)
    return Path(path)
