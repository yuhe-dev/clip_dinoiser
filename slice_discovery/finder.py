from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .types import SliceFindingResult


def _validate_input_matrix(matrix: np.ndarray, sample_ids: list[str], num_slices: int) -> np.ndarray:
    X = np.asarray(matrix, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("matrix must be 2D")
    if X.shape[0] != len(sample_ids):
        raise ValueError("sample_ids length must match matrix rows")
    if X.shape[0] < num_slices:
        raise ValueError("num_slices cannot exceed number of samples")
    return X


def _normalize_rows(matrix: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    safe_matrix = np.nan_to_num(np.asarray(matrix, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(safe_matrix, axis=1, keepdims=True)
    safe_norms = np.clip(norms, epsilon, None)
    normalized = safe_matrix / safe_norms
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def _finalize_result(
    sample_ids: list[str],
    membership: np.ndarray,
    centers: np.ndarray,
    diagnostics: dict[str, object] | None = None,
) -> SliceFindingResult:
    hard_assignment = membership.argmax(axis=1).astype(np.int64)
    slice_weights = membership.mean(axis=0).astype(np.float32)
    return SliceFindingResult(
        sample_ids=list(sample_ids),
        membership=membership.astype(np.float32),
        hard_assignment=hard_assignment,
        slice_weights=slice_weights,
        centers=centers.astype(np.float32),
        diagnostics=dict(diagnostics or {}),
    )


def _reset_dead_components(
    *,
    X: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    weights: np.ndarray,
    responsibilities: np.ndarray,
    dead: np.ndarray,
    rng: np.random.Generator,
    global_var: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not np.any(dead):
        return means, covars, weights, responsibilities

    reinit_indices = rng.choice(X.shape[0], size=int(dead.sum()), replace=False)
    means[dead] = X[reinit_indices]
    covars[dead] = global_var
    weights[dead] = 1.0 / float(X.shape[0])
    responsibilities[:, dead] = 1e-6
    responsibilities = np.nan_to_num(responsibilities, nan=0.0, posinf=0.0, neginf=0.0)
    responsibilities /= np.clip(responsibilities.sum(axis=1, keepdims=True), 1e-12, None)
    return means, covars, weights, responsibilities


def _reset_dead_directional_components(
    *,
    X: np.ndarray,
    means: np.ndarray,
    kappas: np.ndarray,
    weights: np.ndarray,
    responsibilities: np.ndarray,
    dead: np.ndarray,
    rng: np.random.Generator,
    base_kappa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not np.any(dead):
        return means, kappas, weights, responsibilities

    reinit_indices = rng.choice(X.shape[0], size=int(dead.sum()), replace=False)
    means[dead] = X[reinit_indices]
    kappas[dead] = float(base_kappa)
    weights[dead] = 1.0 / float(X.shape[0])
    responsibilities[:, dead] = 1e-6
    responsibilities = np.nan_to_num(responsibilities, nan=0.0, posinf=0.0, neginf=0.0)
    responsibilities /= np.clip(responsibilities.sum(axis=1, keepdims=True), 1e-12, None)
    return means, kappas, weights, responsibilities


class SoftKMeansSliceFinder:
    def __init__(
        self,
        num_slices: int,
        seed: int = 0,
        max_iters: int = 50,
        temperature: float = 1.0,
        tol: float = 1e-5,
    ):
        self.num_slices = int(num_slices)
        self.seed = int(seed)
        self.max_iters = int(max_iters)
        self.temperature = float(temperature)
        self.tol = float(tol)

        if self.num_slices <= 0:
            raise ValueError("num_slices must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    def fit(
        self,
        matrix: np.ndarray,
        sample_ids: list[str],
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> SliceFindingResult:
        X = _validate_input_matrix(matrix, sample_ids, self.num_slices)

        rng = np.random.default_rng(self.seed)
        init_indices = rng.choice(X.shape[0], size=self.num_slices, replace=False)
        centers = X[init_indices].copy()

        membership = np.full((X.shape[0], self.num_slices), 1.0 / self.num_slices, dtype=np.float32)

        for iteration in range(self.max_iters):
            distances = self._squared_distances(X, centers)
            logits = -distances / self.temperature
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            membership = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)

            weights = membership.sum(axis=0, keepdims=True).T
            new_centers = (membership.T @ X) / np.clip(weights, 1e-12, None)
            max_center_delta = float(np.max(np.abs(new_centers - centers)))
            if progress_callback is not None:
                progress_callback(
                    {
                        "finder": "soft_kmeans",
                        "iteration": int(iteration + 1),
                        "max_iters": int(self.max_iters),
                        "max_center_delta": max_center_delta,
                    }
                )

            if max_center_delta <= self.tol:
                centers = new_centers.astype(np.float32)
                break
            centers = new_centers.astype(np.float32)

        row_sums = membership.sum(axis=1)
        diagnostics = {
            "input_shape": [int(X.shape[0]), int(X.shape[1])],
            "input_all_finite": bool(np.isfinite(X).all()),
            "membership_row_sum_min": float(np.min(row_sums)),
            "membership_row_sum_max": float(np.max(row_sums)),
        }
        return _finalize_result(sample_ids, membership, centers, diagnostics=diagnostics)

    @staticmethod
    def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        diffs = X[:, None, :] - centers[None, :, :]
        return np.sum(diffs * diffs, axis=2, dtype=np.float32).astype(np.float32)


class GMMSliceFinder:
    def __init__(
        self,
        num_slices: int,
        seed: int = 0,
        max_iters: int = 100,
        covariance_type: str = "diag",
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
    ):
        self.num_slices = int(num_slices)
        self.seed = int(seed)
        self.max_iters = int(max_iters)
        self.covariance_type = covariance_type
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)

        if self.num_slices <= 0:
            raise ValueError("num_slices must be positive")
        if self.covariance_type != "diag":
            raise ValueError("Only diag covariance_type is currently supported")

    def fit(
        self,
        matrix: np.ndarray,
        sample_ids: list[str],
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> SliceFindingResult:
        X = _validate_input_matrix(matrix, sample_ids, self.num_slices).astype(np.float64, copy=False)
        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.seed)
        init_indices = rng.choice(n_samples, size=self.num_slices, replace=False)
        means = X[init_indices].copy()
        weights = np.full((self.num_slices,), 1.0 / self.num_slices, dtype=np.float64)

        global_var = X.var(axis=0).astype(np.float64) + self.reg_covar
        covars = np.tile(global_var[None, :], (self.num_slices, 1)).astype(np.float64)

        prev_log_likelihood = None
        responsibilities = np.full((n_samples, self.num_slices), 1.0 / self.num_slices, dtype=np.float64)
        min_component_mass = 1e-6
        log_likelihood_trace: list[float] = []

        for iteration in range(self.max_iters):
            log_prob = self._estimate_log_prob(X, means, covars)
            log_weighted = log_prob + np.log(np.clip(weights[None, :], 1e-12, None))
            log_norm = self._logsumexp(log_weighted, axis=1)
            responsibilities = np.exp(log_weighted - log_norm[:, None]).astype(np.float64)

            responsibilities = np.nan_to_num(
                responsibilities,
                nan=1.0 / self.num_slices,
                posinf=1.0 / self.num_slices,
                neginf=0.0,
            )
            responsibilities /= np.clip(responsibilities.sum(axis=1, keepdims=True), 1e-12, None)

            nk = responsibilities.sum(axis=0)
            dead = nk <= min_component_mass
            means, covars, weights, responsibilities = _reset_dead_components(
                X=X,
                means=means,
                covars=covars,
                weights=weights,
                responsibilities=responsibilities,
                dead=dead,
                rng=rng,
                global_var=global_var,
            )
            nk = responsibilities.sum(axis=0)

            nk = nk + 1e-12
            weights = (nk / float(n_samples)).astype(np.float64)
            weighted_sum = responsibilities.T @ X
            means = (weighted_sum / nk[:, None]).astype(np.float64)
            means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)

            for k in range(self.num_slices):
                diff = X - means[k]
                covars[k] = ((responsibilities[:, k][:, None] * (diff * diff)).sum(axis=0) / nk[k]).astype(np.float64)
            covars = np.maximum(covars, self.reg_covar).astype(np.float64)
            covars = np.nan_to_num(covars, nan=self.reg_covar, posinf=self.reg_covar, neginf=self.reg_covar)

            invalid_components = (~np.isfinite(means).all(axis=1)) | (~np.isfinite(covars).all(axis=1)) | (weights <= 0)
            means, covars, weights, responsibilities = _reset_dead_components(
                X=X,
                means=means,
                covars=covars,
                weights=weights,
                responsibilities=responsibilities,
                dead=invalid_components,
                rng=rng,
                global_var=global_var,
            )
            nk = responsibilities.sum(axis=0) + 1e-12
            weights = (nk / float(n_samples)).astype(np.float64)

            log_likelihood = float(log_norm.sum())
            log_likelihood_trace.append(log_likelihood)
            if progress_callback is not None:
                progress_callback(
                    {
                        "finder": "gmm",
                        "iteration": int(iteration + 1),
                        "max_iters": int(self.max_iters),
                        "log_likelihood": log_likelihood,
                    }
                )
            if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) <= self.tol:
                break
            prev_log_likelihood = log_likelihood

        row_sums = responsibilities.sum(axis=1)
        diagnostics = {
            "input_shape": [int(X.shape[0]), int(X.shape[1])],
            "input_all_finite": bool(np.isfinite(X).all()),
            "membership_row_sum_min": float(np.min(row_sums)),
            "membership_row_sum_max": float(np.max(row_sums)),
            "log_likelihood_trace": log_likelihood_trace,
        }
        return _finalize_result(sample_ids, responsibilities, means, diagnostics=diagnostics)

    def _estimate_log_prob(self, X: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
        n_features = X.shape[1]
        log_prob = np.empty((X.shape[0], self.num_slices), dtype=np.float64)
        constant = n_features * np.log(2.0 * np.pi)
        for k in range(self.num_slices):
            safe_covar = np.clip(np.nan_to_num(covars[k], nan=self.reg_covar, posinf=self.reg_covar, neginf=self.reg_covar), self.reg_covar, None)
            safe_mean = np.nan_to_num(means[k], nan=0.0, posinf=0.0, neginf=0.0)
            diff = X - safe_mean
            quadratic = np.sum((diff * diff) / safe_covar[None, :], axis=1, dtype=np.float64)
            log_det = float(np.log(safe_covar).sum())
            log_prob[:, k] = -0.5 * (constant + log_det + quadratic)
        return log_prob

    @staticmethod
    def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
        max_values = np.max(values, axis=axis, keepdims=True)
        stable = values - max_values
        summed = np.sum(np.exp(stable), axis=axis, keepdims=True)
        return (max_values + np.log(np.clip(summed, 1e-12, None))).squeeze(axis)


class VMFSliceFinder:
    def __init__(
        self,
        num_slices: int,
        seed: int = 0,
        max_iters: int = 100,
        tol: float = 1e-5,
        kappa_floor: float = 1e-3,
        kappa_cap: float = 1e4,
    ):
        self.num_slices = int(num_slices)
        self.seed = int(seed)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.kappa_floor = float(kappa_floor)
        self.kappa_cap = float(kappa_cap)

        if self.num_slices <= 0:
            raise ValueError("num_slices must be positive")
        if self.kappa_floor <= 0:
            raise ValueError("kappa_floor must be positive")
        if self.kappa_cap <= self.kappa_floor:
            raise ValueError("kappa_cap must be > kappa_floor")

    def fit(
        self,
        matrix: np.ndarray,
        sample_ids: list[str],
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> SliceFindingResult:
        X = _validate_input_matrix(matrix, sample_ids, self.num_slices).astype(np.float64, copy=False)
        X = _normalize_rows(X)
        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.seed)
        init_indices = rng.choice(n_samples, size=self.num_slices, replace=False)
        means = X[init_indices].copy()
        weights = np.full((self.num_slices,), 1.0 / self.num_slices, dtype=np.float64)

        global_resultant = float(np.linalg.norm(X.mean(axis=0)))
        base_kappa = self._estimate_kappa_from_resultant(global_resultant, n_features)
        kappas = np.full((self.num_slices,), base_kappa, dtype=np.float64)

        prev_log_likelihood = None
        responsibilities = np.full((n_samples, self.num_slices), 1.0 / self.num_slices, dtype=np.float64)
        min_component_mass = 1e-6
        log_likelihood_trace: list[float] = []
        mean_kappa_trace: list[float] = []

        for iteration in range(self.max_iters):
            log_prob = self._estimate_log_prob(X, means, kappas)
            log_weighted = log_prob + np.log(np.clip(weights[None, :], 1e-12, None))
            log_norm = GMMSliceFinder._logsumexp(log_weighted, axis=1)
            responsibilities = np.exp(log_weighted - log_norm[:, None]).astype(np.float64)
            responsibilities = np.nan_to_num(
                responsibilities,
                nan=1.0 / self.num_slices,
                posinf=1.0 / self.num_slices,
                neginf=0.0,
            )
            responsibilities /= np.clip(responsibilities.sum(axis=1, keepdims=True), 1e-12, None)

            nk = responsibilities.sum(axis=0)
            safe_resp = np.nan_to_num(responsibilities, nan=0.0, posinf=0.0, neginf=0.0)
            with np.errstate(all="ignore"):
                resultant = safe_resp.T @ X
            resultant_norm = np.linalg.norm(resultant, axis=1)
            dead = (nk <= min_component_mass) | (resultant_norm <= 1e-12)
            means, kappas, weights, responsibilities = _reset_dead_directional_components(
                X=X,
                means=means,
                kappas=kappas,
                weights=weights,
                responsibilities=responsibilities,
                dead=dead,
                rng=rng,
                base_kappa=base_kappa,
            )

            nk = responsibilities.sum(axis=0) + 1e-12
            weights = (nk / float(n_samples)).astype(np.float64)
            safe_resp = np.nan_to_num(responsibilities, nan=0.0, posinf=0.0, neginf=0.0)
            with np.errstate(all="ignore"):
                resultant = safe_resp.T @ X
            resultant_norm = np.linalg.norm(resultant, axis=1)
            safe_norm = np.clip(resultant_norm, 1e-12, None)
            means = np.nan_to_num(resultant / safe_norm[:, None], nan=0.0, posinf=0.0, neginf=0.0)
            means = _normalize_rows(means)

            rbar = np.clip(resultant_norm / nk, 1e-6, 1.0 - 1e-6)
            kappas = self._estimate_kappa_vector(rbar, n_features)

            invalid = (~np.isfinite(means).all(axis=1)) | (~np.isfinite(kappas)) | (weights <= 0)
            means, kappas, weights, responsibilities = _reset_dead_directional_components(
                X=X,
                means=means,
                kappas=kappas,
                weights=weights,
                responsibilities=responsibilities,
                dead=invalid,
                rng=rng,
                base_kappa=base_kappa,
            )
            nk = responsibilities.sum(axis=0) + 1e-12
            weights = (nk / float(n_samples)).astype(np.float64)

            log_likelihood = float(log_norm.sum())
            log_likelihood_trace.append(log_likelihood)
            mean_kappa_trace.append(float(np.mean(kappas)))
            if progress_callback is not None:
                progress_callback(
                    {
                        "finder": "vmf",
                        "iteration": int(iteration + 1),
                        "max_iters": int(self.max_iters),
                        "log_likelihood": log_likelihood,
                        "mean_kappa": float(np.mean(kappas)),
                    }
                )
            if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) <= self.tol:
                break
            prev_log_likelihood = log_likelihood

        row_sums = responsibilities.sum(axis=1)
        diagnostics = {
            "input_shape": [int(n_samples), int(n_features)],
            "input_all_finite": bool(np.isfinite(X).all()),
            "membership_row_sum_min": float(np.min(row_sums)),
            "membership_row_sum_max": float(np.max(row_sums)),
            "log_likelihood_trace": log_likelihood_trace,
            "mean_kappa_trace": mean_kappa_trace,
            "final_kappas": kappas.astype(float).tolist(),
        }
        return _finalize_result(sample_ids, responsibilities, means, diagnostics=diagnostics)

    def _estimate_log_prob(self, X: np.ndarray, means: np.ndarray, kappas: np.ndarray) -> np.ndarray:
        safe_X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        safe_means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
        with np.errstate(all="ignore"):
            similarities = np.clip(safe_X @ safe_means.T, -1.0, 1.0).astype(np.float64)
        log_c = self._log_vmf_normalizer(kappas, X.shape[1])
        return similarities * kappas[None, :] + log_c[None, :]

    def _estimate_kappa_vector(self, rbar: np.ndarray, n_features: int) -> np.ndarray:
        estimates = np.asarray(
            [self._estimate_kappa_from_resultant(float(value), n_features) for value in rbar],
            dtype=np.float64,
        )
        return np.clip(estimates, self.kappa_floor, self.kappa_cap)

    def _estimate_kappa_from_resultant(self, rbar: float, n_features: int) -> float:
        clipped = min(max(float(rbar), 1e-6), 1.0 - 1e-6)
        estimate = clipped * (float(n_features) - clipped * clipped) / max(1e-6, 1.0 - clipped * clipped)
        if not np.isfinite(estimate):
            return float(self.kappa_floor)
        return float(np.clip(estimate, self.kappa_floor, self.kappa_cap))

    @staticmethod
    def _log_vmf_normalizer(kappas: np.ndarray, n_features: int) -> np.ndarray:
        kappas = np.asarray(kappas, dtype=np.float64)
        p = float(n_features)
        half_p = 0.5 * p
        nu = half_p - 1.0
        uniform_log_c = math.lgamma(half_p) - math.log(2.0) - half_p * math.log(math.pi)
        output = np.full_like(kappas, uniform_log_c, dtype=np.float64)
        if nu <= 0.0:
            return output

        mask = kappas >= 1e-3
        if not np.any(mask):
            return output

        active = np.clip(kappas[mask], 1e-6, None)
        z = active / nu
        sqrt_term = np.sqrt(1.0 + z * z)
        eta = sqrt_term + np.log(np.clip(z / (1.0 + sqrt_term), 1e-12, None))
        log_iv = -0.5 * np.log(2.0 * math.pi * nu) - 0.25 * np.log(1.0 + z * z) + nu * eta
        output[mask] = nu * np.log(active) - half_p * math.log(2.0 * math.pi) - log_iv
        return np.nan_to_num(output, nan=uniform_log_c, posinf=uniform_log_c, neginf=uniform_log_c)
