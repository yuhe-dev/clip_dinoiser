"""Queue scheduling helpers for autonomous research loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from .contracts import ExperimentCard
from .registry import load_experiment_card, write_json
from .runtime import is_utc_iso_stale, utc_now_iso


QUEUEABLE_STATUSES = {"queued", "stale_requeued"}
BUDGET_TIER_RANK = {"Tier A": 0, "Tier B": 1, "Tier C": 2}


def list_experiment_cards(scan_dir: str | Path) -> List[Tuple[Path, ExperimentCard]]:
    root = Path(scan_dir)
    if not root.exists():
        return []
    return [(candidate, load_experiment_card(candidate)) for candidate in sorted(root.glob("*.json"))]


def dependency_status_map(cards: List[Tuple[Path, ExperimentCard]]) -> Dict[str, str]:
    return {card.experiment_id: card.status for _, card in cards}


def card_wait_reason(
    card: ExperimentCard,
    *,
    status_by_id: Dict[str, str],
) -> str:
    if card.status not in QUEUEABLE_STATUSES:
        return f"status={card.status}"
    if card.next_eligible_at_utc and not is_utc_iso_stale(card.next_eligible_at_utc, ttl_seconds=0):
        return f"next_eligible_at_utc={card.next_eligible_at_utc}"
    if int(card.max_attempts) > 0 and int(card.attempt_count) >= int(card.max_attempts):
        return f"retry_limit_reached:{card.attempt_count}/{card.max_attempts}"
    for dependency in card.depends_on:
        dep_status = status_by_id.get(dependency, "missing")
        if dep_status != "completed":
            return f"waiting_dependency:{dependency}:{dep_status}"
    return ""


def ready_cards(cards: List[Tuple[Path, ExperimentCard]]) -> List[Tuple[Path, ExperimentCard]]:
    status_by_id = dependency_status_map(cards)
    return [
        (path, card)
        for path, card in cards
        if not card_wait_reason(card, status_by_id=status_by_id)
    ]


def scheduler_sort_key(item: Tuple[Path, ExperimentCard]) -> tuple[Any, ...]:
    path, card = item
    return (
        int(card.priority),
        BUDGET_TIER_RANK.get(card.budget_tier, 99),
        int(card.attempt_count),
        card.experiment_id,
        str(path),
    )


def select_experiment_card(scan_dir: str | Path) -> Path | None:
    ready = ready_cards(list_experiment_cards(scan_dir))
    if not ready:
        return None
    path, _ = sorted(ready, key=scheduler_sort_key)[0]
    return path


def build_queue_snapshot(scan_dir: str | Path) -> Dict[str, Any]:
    cards = list_experiment_cards(scan_dir)
    status_by_id = dependency_status_map(cards)
    ready = []
    waiting = []
    terminal = []
    for path, card in cards:
        reason = card_wait_reason(card, status_by_id=status_by_id)
        entry = {
            "experiment_id": card.experiment_id,
            "status": card.status,
            "priority": card.priority,
            "budget_tier": card.budget_tier,
            "attempt_count": card.attempt_count,
            "card_path": str(path.resolve()),
        }
        if reason:
            entry["wait_reason"] = reason
        if card.status in QUEUEABLE_STATUSES and not reason:
            ready.append(entry)
        elif card.status in QUEUEABLE_STATUSES:
            waiting.append(entry)
        else:
            terminal.append(entry)
    ready_sorted = sorted(ready, key=lambda item: (item["priority"], BUDGET_TIER_RANK.get(item["budget_tier"], 99), item["attempt_count"], item["experiment_id"]))
    return {
        "generated_at_utc": utc_now_iso(),
        "total_cards": len(cards),
        "ready_count": len(ready_sorted),
        "waiting_count": len(waiting),
        "terminal_count": len(terminal),
        "ready": ready_sorted,
        "waiting": waiting,
        "terminal": terminal,
    }


def write_queue_snapshot(path: str | Path, payload: Dict[str, Any]) -> Path:
    write_json(path, payload)
    return Path(path)
