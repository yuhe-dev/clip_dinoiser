from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .finder import GMMSliceFinder, VMFSliceFinder


@dataclass
class SliceSelectionThresholds:
    min_slice_weight: float = 0.005
    min_hard_count: int = 250
    min_avg_max_membership: float = 0.90
    max_avg_entropy: float = 0.20
    min_coherence: float = 0.47
    bic_relative_tolerance: float = 0.05
    coherence_relative_tolerance: float = 0.05
    log_likelihood_relative_tolerance: float = 0.05


@dataclass
class SliceSelectionCandidate:
    num_slices: int
    bic: float
    log_likelihood: float
    min_slice_weight: float
    min_hard_count: int
    avg_max_membership: float
    avg_entropy: float
    mean_coherence: float
    min_coherence: float
    admissible: bool
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def generate_candidate_ks(min_num_slices: int, max_num_slices: int) -> list[int]:
    lower = int(min_num_slices)
    upper = int(max_num_slices)
    if lower <= 0:
        raise ValueError("min_num_slices must be positive")
    if upper < lower:
        raise ValueError("max_num_slices must be >= min_num_slices")

    candidates: set[int] = {lower, upper}
    power = 1
    while power < lower:
        power *= 2

    prev: int | None = None
    while power <= upper:
        candidates.add(power)
        if prev is not None:
            midpoint = (prev + power) // 2
            if lower <= midpoint <= upper:
                candidates.add(midpoint)
        prev = power
        power *= 2

    return sorted(candidate for candidate in candidates if lower <= candidate <= upper)


def compute_gmm_bic(*, log_likelihood: float, n_samples: int, n_features: int, num_slices: int) -> float:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if num_slices <= 0:
        raise ValueError("num_slices must be positive")

    parameter_count = int(num_slices) * int(n_features) * 2 + (int(num_slices) - 1)
    return float(-2.0 * float(log_likelihood) + parameter_count * np.log(float(n_samples)))


def compute_vmf_bic(*, log_likelihood: float, n_samples: int, n_features: int, num_slices: int) -> float:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if num_slices <= 0:
        raise ValueError("num_slices must be positive")

    # Mixture weights: K - 1
    # Mean directions on S^(D-1): K * (D - 1)
    # Per-component concentration kappa: K
    parameter_count = int(num_slices) * int(n_features) + (int(num_slices) - 1)
    return float(-2.0 * float(log_likelihood) + parameter_count * np.log(float(n_samples)))


def _compute_slice_coherence(
    matrix: np.ndarray,
    membership: np.ndarray,
    centers: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    vectors = np.asarray(matrix, dtype=np.float64)
    weights = np.asarray(membership, dtype=np.float64)
    centroid_matrix = np.asarray(centers, dtype=np.float64)

    sample_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    center_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True).T
    denom = np.clip(sample_norms * center_norms, epsilon, None)
    safe_vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    safe_centers = np.nan_to_num(centroid_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    with np.errstate(all="ignore"):
        similarities = np.clip(
            (safe_vectors @ safe_centers.T) / np.nan_to_num(denom, nan=epsilon, posinf=epsilon, neginf=epsilon),
            -1.0,
            1.0,
        )
    similarities = np.nan_to_num(similarities, nan=0.0, posinf=0.0, neginf=0.0)

    slice_mass = np.clip(weights.sum(axis=0), epsilon, None)
    return ((weights * similarities).sum(axis=0) / slice_mass).astype(np.float64)


def _build_candidate_from_result(
    *,
    matrix: np.ndarray,
    result,
    num_slices: int,
    thresholds: SliceSelectionThresholds,
    bic: float,
    log_likelihood_key: str,
) -> SliceSelectionCandidate:
    membership = np.asarray(result.membership, dtype=np.float64)
    slice_weights = np.asarray(result.slice_weights, dtype=np.float64)
    hard_counts = np.bincount(result.hard_assignment, minlength=int(num_slices)).astype(np.int64)
    max_membership = membership.max(axis=1)
    entropy = -(membership * np.log(np.clip(membership, 1e-12, None))).sum(axis=1)
    coherence = _compute_slice_coherence(matrix, membership, result.centers)

    log_likelihood_trace = list(result.diagnostics.get(log_likelihood_key, []) if result.diagnostics else [])
    if not log_likelihood_trace:
        raise ValueError(f"finder diagnostics must include {log_likelihood_key} for model selection")
    final_log_likelihood = float(log_likelihood_trace[-1])
    if not np.isfinite(final_log_likelihood):
        final_log_likelihood = -1e18

    rejection_reasons: list[str] = []
    min_slice_weight = float(slice_weights.min()) if slice_weights.size else 0.0
    min_hard_count = int(hard_counts.min()) if hard_counts.size else 0
    avg_max_membership = float(max_membership.mean()) if max_membership.size else 0.0
    avg_entropy = float(entropy.mean()) if entropy.size else 0.0
    min_coherence = float(coherence.min()) if coherence.size else 0.0
    mean_coherence = float(coherence.mean()) if coherence.size else 0.0

    if not np.isfinite(min_slice_weight):
        rejection_reasons.append("nonfinite_slice_weight")
    if not np.isfinite(avg_max_membership):
        rejection_reasons.append("nonfinite_avg_max_membership")
    if not np.isfinite(avg_entropy):
        rejection_reasons.append("nonfinite_avg_entropy")
    if not np.isfinite(min_coherence):
        rejection_reasons.append("nonfinite_min_coherence")
    if min_slice_weight < float(thresholds.min_slice_weight):
        rejection_reasons.append("min_slice_weight")
    if min_hard_count < int(thresholds.min_hard_count):
        rejection_reasons.append("min_hard_count")
    if avg_max_membership < float(thresholds.min_avg_max_membership):
        rejection_reasons.append("avg_max_membership")
    if avg_entropy > float(thresholds.max_avg_entropy):
        rejection_reasons.append("avg_entropy")
    if min_coherence < float(thresholds.min_coherence):
        rejection_reasons.append("min_coherence")

    return SliceSelectionCandidate(
        num_slices=int(num_slices),
        bic=float(bic),
        log_likelihood=final_log_likelihood,
        min_slice_weight=min_slice_weight,
        min_hard_count=min_hard_count,
        avg_max_membership=avg_max_membership,
        avg_entropy=avg_entropy,
        mean_coherence=mean_coherence,
        min_coherence=min_coherence,
        admissible=not rejection_reasons,
        rejection_reasons=rejection_reasons,
    )


def evaluate_gmm_candidate(
    *,
    matrix: np.ndarray,
    sample_ids: list[str],
    num_slices: int,
    thresholds: SliceSelectionThresholds,
    seed: int = 0,
    max_iters: int = 100,
    ) -> SliceSelectionCandidate:
    finder = GMMSliceFinder(
        num_slices=int(num_slices),
        seed=int(seed),
        max_iters=int(max_iters),
        covariance_type="diag",
    )
    result = finder.fit(matrix, sample_ids)

    log_likelihood_trace = list(result.diagnostics.get("log_likelihood_trace", []) if result.diagnostics else [])
    if not log_likelihood_trace:
        raise ValueError("GMM diagnostics must include log_likelihood_trace for model selection")
    final_log_likelihood = float(log_likelihood_trace[-1])
    if not np.isfinite(final_log_likelihood):
        final_log_likelihood = -1e18
    bic = compute_gmm_bic(
        log_likelihood=final_log_likelihood,
        n_samples=int(np.asarray(matrix).shape[0]),
        n_features=int(np.asarray(matrix).shape[1]),
        num_slices=int(num_slices),
    )
    return _build_candidate_from_result(
        matrix=matrix,
        result=result,
        num_slices=int(num_slices),
        thresholds=thresholds,
        bic=bic,
        log_likelihood_key="log_likelihood_trace",
    )

def evaluate_vmf_candidate(
    *,
    matrix: np.ndarray,
    sample_ids: list[str],
    num_slices: int,
    thresholds: SliceSelectionThresholds,
    seed: int = 0,
    max_iters: int = 100,
) -> SliceSelectionCandidate:
    finder = VMFSliceFinder(
        num_slices=int(num_slices),
        seed=int(seed),
        max_iters=int(max_iters),
    )
    result = finder.fit(matrix, sample_ids)

    log_likelihood_trace = list(result.diagnostics.get("log_likelihood_trace", []) if result.diagnostics else [])
    if not log_likelihood_trace:
        raise ValueError("vMF diagnostics must include log_likelihood_trace for model selection")
    final_log_likelihood = float(log_likelihood_trace[-1])
    if not np.isfinite(final_log_likelihood):
        final_log_likelihood = -1e18
    bic = compute_vmf_bic(
        log_likelihood=final_log_likelihood,
        n_samples=int(np.asarray(matrix).shape[0]),
        n_features=int(np.asarray(matrix).shape[1]),
        num_slices=int(num_slices),
    )
    return _build_candidate_from_result(
        matrix=matrix,
        result=result,
        num_slices=int(num_slices),
        thresholds=thresholds,
        bic=bic,
        log_likelihood_key="log_likelihood_trace",
    )


def _is_near_best_higher_is_better(value: float, best: float, tolerance: float) -> bool:
    slack = max(abs(float(best)), 1.0) * float(tolerance)
    return float(value) >= float(best) - slack


def select_best_candidate(
    candidates: list[SliceSelectionCandidate],
    thresholds: SliceSelectionThresholds,
    *,
    finder: str = "gmm",
) -> SliceSelectionCandidate:
    if not candidates:
        raise ValueError("candidates must not be empty")

    admissible = [candidate for candidate in candidates if candidate.admissible]
    if finder == "vmf":
        if admissible:
            best_mean_coherence = max(candidate.mean_coherence for candidate in admissible)
            best_min_coherence = max(candidate.min_coherence for candidate in admissible)
            best_log_likelihood = max(candidate.log_likelihood for candidate in admissible)
            near_plateau = [
                candidate
                for candidate in admissible
                if _is_near_best_higher_is_better(
                    candidate.mean_coherence,
                    best_mean_coherence,
                    thresholds.coherence_relative_tolerance,
                )
                and _is_near_best_higher_is_better(
                    candidate.min_coherence,
                    best_min_coherence,
                    thresholds.coherence_relative_tolerance,
                )
                and _is_near_best_higher_is_better(
                    candidate.log_likelihood,
                    best_log_likelihood,
                    thresholds.log_likelihood_relative_tolerance,
                )
            ]
            return min(
                near_plateau,
                key=lambda candidate: (
                    candidate.num_slices,
                    -candidate.mean_coherence,
                    -candidate.min_coherence,
                    -candidate.log_likelihood,
                ),
            )

        return min(
            candidates,
            key=lambda candidate: (
                len(candidate.rejection_reasons),
                -candidate.min_coherence,
                -candidate.mean_coherence,
                -candidate.log_likelihood,
                candidate.num_slices,
            ),
        )

    if admissible:
        best_bic = min(candidate.bic for candidate in admissible)
        bic_window = best_bic + abs(best_bic) * float(thresholds.bic_relative_tolerance)
        near_best = [candidate for candidate in admissible if candidate.bic <= bic_window]
        return min(
            near_best,
            key=lambda candidate: (
                candidate.num_slices,
                candidate.bic,
                -candidate.mean_coherence,
                -candidate.min_coherence,
            ),
        )

    return min(
        candidates,
        key=lambda candidate: (
            len(candidate.rejection_reasons),
            -candidate.min_coherence,
            -candidate.mean_coherence,
            candidate.num_slices,
            candidate.bic,
        ),
    )
