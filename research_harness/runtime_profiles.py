"""Runtime profile registry helpers for worker selection."""

from __future__ import annotations

from typing import Any, Dict, List

from .contracts import ExperimentCard
from .registry import load_json


def load_runtime_profiles(path: str) -> Dict[str, Any]:
    payload = load_json(path)
    payload.setdefault("default_profile_candidates", [])
    payload.setdefault("profiles", {})
    return payload


def resolve_runtime_profile_candidates(
    card: ExperimentCard,
    policy: Dict[str, Any],
    registry: Dict[str, Any],
) -> List[str]:
    metadata = dict(card.metadata)

    explicit = str(metadata.get("runtime_profile_id", "")).strip()
    if explicit:
        return [explicit]

    explicit_candidates = [
        str(candidate).strip()
        for candidate in metadata.get("runtime_profile_candidates", [])
        if str(candidate).strip()
    ]
    if explicit_candidates:
        return explicit_candidates

    policy_default = str(policy.get("default_runtime_profile", "")).strip()
    if policy_default:
        return [policy_default]

    policy_candidates = [
        str(candidate).strip()
        for candidate in policy.get("default_runtime_profile_candidates", [])
        if str(candidate).strip()
    ]
    if policy_candidates:
        return policy_candidates

    registry_candidates = [
        str(candidate).strip()
        for candidate in registry.get("default_profile_candidates", [])
        if str(candidate).strip()
    ]
    if registry_candidates:
        return registry_candidates

    return []
