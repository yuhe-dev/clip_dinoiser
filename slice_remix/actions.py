from __future__ import annotations

import numpy as np

from .types import CandidateAction


def _build_candidate(
    baseline: np.ndarray,
    delta: np.ndarray,
    donors: list[int],
    receivers: list[int],
    amplitude: float,
) -> CandidateAction:
    target = baseline + delta
    if np.any(target < -1e-8):
        raise ValueError("target mixture must remain non-negative")
    if not np.isclose(float(target.sum()), 1.0, atol=1e-5):
        raise ValueError("target mixture must remain on the simplex")

    support_size = int(np.count_nonzero(np.abs(delta) > 1e-8))
    return CandidateAction(
        baseline_mixture=baseline.astype(np.float32).tolist(),
        target_mixture=target.astype(np.float32).tolist(),
        delta_q=delta.astype(np.float32).tolist(),
        donors=list(donors),
        receivers=list(receivers),
        amplitude=float(amplitude),
        support_size=support_size,
        metadata={},
    )


def generate_pairwise_candidates(
    baseline_mixture: np.ndarray,
    amplitudes: list[float],
    ordered_pairs: list[tuple[int, int]],
) -> list[CandidateAction]:
    baseline = np.asarray(baseline_mixture, dtype=np.float32)
    if baseline.ndim != 1:
        raise ValueError("baseline_mixture must be a 1D array")
    if not np.isclose(float(baseline.sum()), 1.0, atol=1e-5):
        raise ValueError("baseline_mixture must sum to 1")

    candidates: list[CandidateAction] = []
    for receiver, donor in ordered_pairs:
        for amplitude in amplitudes:
            tau = float(amplitude)
            if tau < 0.0:
                raise ValueError("amplitude must be non-negative")
            if baseline[donor] + 1e-8 < tau:
                continue
            if baseline[receiver] + tau > 1.0 + 1e-8:
                continue

            delta = np.zeros_like(baseline, dtype=np.float32)
            delta[receiver] += tau
            delta[donor] -= tau
            candidates.append(
                _build_candidate(
                    baseline=baseline,
                    delta=delta,
                    donors=[int(donor)],
                    receivers=[int(receiver)],
                    amplitude=tau,
                )
            )
    return candidates


def select_pairwise_directions(
    *,
    baseline_mixture: np.ndarray,
    portraits: dict[str, np.ndarray],
    max_pairs: int,
    ordered_pairs: list[tuple[int, int]],
    min_amplitude: float = 0.0,
    diversity_weight: float = 1.0,
    reuse_penalty: float = 0.85,
    reverse_pair_penalty: float = 0.5,
) -> list[tuple[int, int]]:
    baseline = np.asarray(baseline_mixture, dtype=np.float32)
    if baseline.ndim != 1:
        raise ValueError("baseline_mixture must be a 1D array")
    if max_pairs <= 0:
        return []

    candidate_specs: list[dict[str, object]] = []
    for receiver, donor in ordered_pairs:
        if receiver == donor:
            continue
        if baseline[donor] + 1e-8 < float(min_amplitude):
            continue
        direction, magnitude = _pair_direction(portraits, int(receiver), int(donor))
        if magnitude <= 0.0:
            continue
        candidate_specs.append(
            {
                "pair": (int(receiver), int(donor)),
                "direction": direction,
                "magnitude": float(magnitude),
            }
        )

    if not candidate_specs:
        return []

    selected: list[dict[str, object]] = []
    while candidate_specs and len(selected) < max_pairs:
        if not selected:
            best = max(candidate_specs, key=lambda spec: (float(spec["magnitude"]), spec["pair"]))
        else:
            best = max(
                candidate_specs,
                key=lambda spec: (
                    _selection_score(
                        spec=spec,
                        selected=selected,
                        diversity_weight=float(diversity_weight),
                        reuse_penalty=float(reuse_penalty),
                        reverse_pair_penalty=float(reverse_pair_penalty),
                    ),
                    spec["pair"],
                ),
            )
        selected.append(best)
        candidate_specs = [spec for spec in candidate_specs if spec["pair"] != best["pair"]]

    return [tuple(spec["pair"]) for spec in selected]


def _pair_direction(
    portraits: dict[str, np.ndarray],
    receiver: int,
    donor: int,
) -> tuple[np.ndarray, float]:
    balanced_parts: list[np.ndarray] = []
    magnitude = 0.0
    for name in sorted(portraits.keys()):
        portrait_matrix = np.asarray(portraits[name], dtype=np.float32)
        delta = np.asarray(portrait_matrix[receiver] - portrait_matrix[donor], dtype=np.float32).reshape(-1)
        group_norm = float(np.linalg.norm(delta))
        magnitude += group_norm
        if group_norm > 1e-8:
            balanced_parts.append(delta / group_norm)
        else:
            balanced_parts.append(np.zeros_like(delta))
    return np.concatenate(balanced_parts, axis=0), magnitude


def _selection_score(
    *,
    spec: dict[str, object],
    selected: list[dict[str, object]],
    diversity_weight: float,
    reuse_penalty: float,
    reverse_pair_penalty: float,
) -> float:
    pair = tuple(spec["pair"])
    direction = np.asarray(spec["direction"], dtype=np.float32)
    magnitude = float(spec["magnitude"])
    diversity = min(_cosine_distance(direction, np.asarray(picked["direction"], dtype=np.float32)) for picked in selected)

    role_penalty = 1.0
    receiver, donor = pair
    for picked in selected:
        picked_receiver, picked_donor = tuple(picked["pair"])
        if picked_receiver == receiver or picked_donor == donor:
            role_penalty *= reuse_penalty
        if picked_receiver == donor and picked_donor == receiver:
            role_penalty *= reverse_pair_penalty

    return magnitude * (1.0 + diversity_weight * diversity) * role_penalty


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return 0.0
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    cosine = max(-1.0, min(1.0, cosine))
    return 1.0 - cosine
