from __future__ import annotations

import copy
from typing import Any

import numpy as np


class LinearRemixSurrogate:
    def __init__(self, ridge: float = 1e-4):
        self.ridge = float(ridge)
        self.delta_q_dim_: int | None = None
        self.feature_order_: list[str] = []
        self.feature_dims_: dict[str, int] = {}
        self.context_keys_: list[str] = []
        self.coef_: np.ndarray | None = None

    def fit(self, rows: list[dict[str, Any]]) -> "LinearRemixSurrogate":
        if not rows:
            raise ValueError("rows must not be empty")

        self.delta_q_dim_ = len(rows[0]["delta_q"])
        self.feature_order_ = sorted(rows[0].get("delta_phi", {}).keys())
        self.feature_dims_ = {
            name: len(rows[0]["delta_phi"][name])
            for name in self.feature_order_
        }
        self.context_keys_ = sorted(
            key
            for key, value in rows[0].get("context", {}).items()
            if isinstance(value, (int, float, bool))
        )

        X = np.asarray([self._row_to_vector(row) for row in rows], dtype=np.float64)
        y = np.asarray([float(row["measured_gain"]) for row in rows], dtype=np.float64)

        ones = np.ones((X.shape[0], 1), dtype=np.float64)
        design = np.concatenate([ones, X], axis=1)
        gram = design.T @ design
        gram += self.ridge * np.eye(gram.shape[0], dtype=np.float64)
        rhs = design.T @ y
        self.coef_ = np.linalg.solve(gram, rhs)
        return self

    def predict_mean(self, rows: list[dict[str, Any]]) -> list[float]:
        if self.coef_ is None:
            raise ValueError("model must be fit before prediction")

        X = np.asarray([self._row_to_vector(row) for row in rows], dtype=np.float64)
        ones = np.ones((X.shape[0], 1), dtype=np.float64)
        design = np.concatenate([ones, X], axis=1)
        preds = design @ self.coef_
        return [float(value) for value in preds.tolist()]

    def predict_std(self, rows: list[dict[str, Any]]) -> list[float]:
        return [0.0 for _ in rows]

    def _row_to_vector(self, row: dict[str, Any]) -> list[float]:
        if self.delta_q_dim_ is None:
            raise ValueError("model metadata is not initialized")

        vector: list[float] = []
        delta_q = row.get("delta_q", [])
        if len(delta_q) != self.delta_q_dim_:
            raise ValueError("delta_q dimension mismatch")
        vector.extend(self._encode_delta_q(delta_q))

        delta_phi = row.get("delta_phi", {})
        for name in self.feature_order_:
            values = delta_phi.get(name, [0.0] * self.feature_dims_[name])
            if len(values) != self.feature_dims_[name]:
                raise ValueError(f"delta_phi dimension mismatch for feature group '{name}'")
            vector.extend(float(value) for value in values)

        context = row.get("context", {})
        for key in self.context_keys_:
            value = context.get(key, 0.0)
            if not isinstance(value, (int, float, bool)):
                raise ValueError(f"context key '{key}' must be numeric")
            vector.append(float(value))

        return vector

    def _encode_delta_q(self, delta_q: list[float]) -> list[float]:
        return [float(value) for value in delta_q]


class QuadraticRemixSurrogate(LinearRemixSurrogate):
    def _encode_delta_q(self, delta_q: list[float]) -> list[float]:
        values = [float(value) for value in delta_q]
        encoded = list(values)
        for left_index, left_value in enumerate(values):
            for right_value in values[left_index:]:
                encoded.append(left_value * right_value)
        return encoded


class BootstrapRemixSurrogate:
    def __init__(self, base_model: str = "linear", num_models: int = 8, seed: int = 0):
        if num_models <= 0:
            raise ValueError("num_models must be positive")
        self.base_model = str(base_model)
        self.num_models = int(num_models)
        self.seed = int(seed)
        self.models_: list[LinearRemixSurrogate] = []

    def fit(self, rows: list[dict[str, Any]]) -> "BootstrapRemixSurrogate":
        if not rows:
            raise ValueError("rows must not be empty")

        rng = np.random.default_rng(self.seed)
        self.models_ = []
        row_count = len(rows)
        for _ in range(self.num_models):
            sample_indices = rng.integers(0, row_count, size=row_count)
            sampled_rows = [copy.deepcopy(rows[int(index)]) for index in sample_indices.tolist()]
            self.models_.append(_make_base_surrogate(self.base_model).fit(sampled_rows))
        return self

    def predict_mean(self, rows: list[dict[str, Any]]) -> list[float]:
        means, _ = self._predict_ensemble(rows)
        return means

    def predict_std(self, rows: list[dict[str, Any]]) -> list[float]:
        _, stds = self._predict_ensemble(rows)
        return stds

    def _predict_ensemble(self, rows: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
        if not self.models_:
            raise ValueError("model must be fit before prediction")
        ensemble = np.asarray([model.predict_mean(rows) for model in self.models_], dtype=np.float64)
        mean = ensemble.mean(axis=0)
        std = ensemble.std(axis=0, ddof=0)
        return mean.astype(float).tolist(), std.astype(float).tolist()


def _make_base_surrogate(model_name: str) -> LinearRemixSurrogate:
    if model_name == "linear":
        return LinearRemixSurrogate()
    if model_name == "quadratic":
        return QuadraticRemixSurrogate()
    raise ValueError(f"Unsupported base_model='{model_name}'")
