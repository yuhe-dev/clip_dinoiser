from __future__ import annotations

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


def _finalize_result(sample_ids: list[str], membership: np.ndarray, centers: np.ndarray) -> SliceFindingResult:
    hard_assignment = membership.argmax(axis=1).astype(np.int64)
    slice_weights = membership.mean(axis=0).astype(np.float32)
    return SliceFindingResult(
        sample_ids=list(sample_ids),
        membership=membership.astype(np.float32),
        hard_assignment=hard_assignment,
        slice_weights=slice_weights,
        centers=centers.astype(np.float32),
    )


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

    def fit(self, matrix: np.ndarray, sample_ids: list[str]) -> SliceFindingResult:
        X = _validate_input_matrix(matrix, sample_ids, self.num_slices)

        rng = np.random.default_rng(self.seed)
        init_indices = rng.choice(X.shape[0], size=self.num_slices, replace=False)
        centers = X[init_indices].copy()

        membership = np.full((X.shape[0], self.num_slices), 1.0 / self.num_slices, dtype=np.float32)

        for _ in range(self.max_iters):
            distances = self._squared_distances(X, centers)
            logits = -distances / self.temperature
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            membership = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)

            weights = membership.sum(axis=0, keepdims=True).T
            new_centers = (membership.T @ X) / np.clip(weights, 1e-12, None)

            if np.max(np.abs(new_centers - centers)) <= self.tol:
                centers = new_centers.astype(np.float32)
                break
            centers = new_centers.astype(np.float32)

        return _finalize_result(sample_ids, membership, centers)

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

    def fit(self, matrix: np.ndarray, sample_ids: list[str]) -> SliceFindingResult:
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

        for _ in range(self.max_iters):
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
            if np.any(dead):
                reinit_indices = rng.choice(n_samples, size=int(dead.sum()), replace=False)
                means[dead] = X[reinit_indices]
                covars[dead] = global_var
                weights[dead] = 1.0 / n_samples
                responsibilities[:, dead] = 1e-6
                responsibilities /= np.clip(responsibilities.sum(axis=1, keepdims=True), 1e-12, None)
                nk = responsibilities.sum(axis=0)

            nk = nk + 1e-12
            weights = (nk / float(n_samples)).astype(np.float64)
            weighted_sum = responsibilities.T @ X
            means = (weighted_sum / nk[:, None]).astype(np.float64)

            for k in range(self.num_slices):
                diff = X - means[k]
                covars[k] = ((responsibilities[:, k][:, None] * (diff * diff)).sum(axis=0) / nk[k]).astype(np.float64)
            covars = np.maximum(covars, self.reg_covar).astype(np.float64)

            log_likelihood = float(log_norm.sum())
            if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) <= self.tol:
                break
            prev_log_likelihood = log_likelihood

        return _finalize_result(sample_ids, responsibilities, means)

    def _estimate_log_prob(self, X: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
        n_features = X.shape[1]
        log_prob = np.empty((X.shape[0], self.num_slices), dtype=np.float64)
        constant = n_features * np.log(2.0 * np.pi)
        for k in range(self.num_slices):
            safe_covar = np.clip(covars[k], self.reg_covar, None)
            diff = X - means[k]
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
