from __future__ import annotations

import copy
import json
import math
import os
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

    def evaluate(
        self,
        rows: list[dict[str, Any]],
        *,
        group_key: str = "baseline_seed",
        top_k: int = 3,
        kappa: float = 0.0,
    ) -> dict[str, Any]:
        return evaluate_surrogate_predictions(
            rows,
            predicted_mean=self.predict_mean(rows),
            predicted_std=self.predict_std(rows),
            group_key=group_key,
            top_k=top_k,
            kappa=kappa,
        )

    def to_payload(self) -> dict[str, Any]:
        if self.coef_ is None:
            raise ValueError("model must be fit before serialization")
        return {
            "type": self.__class__.__name__,
            "algorithm": describe_surrogate_algorithm("linear", bootstrap_models=1),
            "ridge": self.ridge,
            "delta_q_dim": self.delta_q_dim_,
            "feature_order": list(self.feature_order_),
            "feature_dims": dict(self.feature_dims_),
            "context_keys": list(self.context_keys_),
            "coef": self.coef_.astype(float).tolist(),
        }

    def save_json(self, path: str) -> str:
        output_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_payload(), f, indent=2, ensure_ascii=False)
        return output_path

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
    def to_payload(self) -> dict[str, Any]:
        payload = super().to_payload()
        payload["algorithm"] = describe_surrogate_algorithm("quadratic", bootstrap_models=1)
        return payload

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

    def evaluate(
        self,
        rows: list[dict[str, Any]],
        *,
        group_key: str = "baseline_seed",
        top_k: int = 3,
        kappa: float = 0.0,
    ) -> dict[str, Any]:
        return evaluate_surrogate_predictions(
            rows,
            predicted_mean=self.predict_mean(rows),
            predicted_std=self.predict_std(rows),
            group_key=group_key,
            top_k=top_k,
            kappa=kappa,
        )

    def to_payload(self) -> dict[str, Any]:
        if not self.models_:
            raise ValueError("model must be fit before serialization")
        return {
            "type": self.__class__.__name__,
            "algorithm": describe_surrogate_algorithm(self.base_model, bootstrap_models=self.num_models),
            "base_model": self.base_model,
            "num_models": self.num_models,
            "seed": self.seed,
            "models": [model.to_payload() for model in self.models_],
        }

    def save_json(self, path: str) -> str:
        output_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_payload(), f, indent=2, ensure_ascii=False)
        return output_path

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


def build_surrogate(model_name: str, bootstrap_models: int = 1):
    if bootstrap_models > 1:
        return BootstrapRemixSurrogate(base_model=model_name, num_models=bootstrap_models)
    return _make_base_surrogate(model_name)


def describe_surrogate_algorithm(model_name: str, *, bootstrap_models: int = 1) -> dict[str, Any]:
    algorithm = {
        "base_model": model_name,
        "regularization": "ridge_l2",
        "delta_q_encoding": "linear" if model_name == "linear" else "quadratic_upper_triangle",
        "delta_phi_encoding": "grouped_linear_concat",
        "context_encoding": "numeric_linear_concat",
    }
    if bootstrap_models > 1:
        algorithm["ensemble"] = {
            "type": "bootstrap",
            "num_models": int(bootstrap_models),
        }
    else:
        algorithm["ensemble"] = {
            "type": "none",
            "num_models": 1,
        }
    return algorithm


def _resolve_group_id(row: dict[str, Any], group_key: str) -> str:
    context = row.get("context", {})
    if isinstance(context, dict) and group_key in context:
        return str(context[group_key])
    if group_key in row:
        return str(row[group_key])
    return "default"


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _rankdata(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start
        while end + 1 < len(array) and array[order[end + 1]] == array[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end) + 1.0
        ranks[order[start : end + 1]] = average_rank
        start = end + 1
    return ranks


def _spearman_correlation(actual: list[float], predicted: list[float]) -> float | None:
    if len(actual) < 2:
        return None
    rank_actual = _rankdata(actual)
    rank_predicted = _rankdata(predicted)
    if np.std(rank_actual) == 0.0 or np.std(rank_predicted) == 0.0:
        return None
    correlation = np.corrcoef(rank_actual, rank_predicted)[0, 1]
    return float(correlation)


def evaluate_surrogate_predictions(
    rows: list[dict[str, Any]],
    *,
    predicted_mean: list[float],
    predicted_std: list[float] | None = None,
    group_key: str = "baseline_seed",
    top_k: int = 3,
    kappa: float = 0.0,
) -> dict[str, Any]:
    if len(rows) != len(predicted_mean):
        raise ValueError("rows and predicted_mean must have the same length")
    if predicted_std is None:
        predicted_std = [0.0 for _ in rows]
    if len(rows) != len(predicted_std):
        raise ValueError("rows and predicted_std must have the same length")
    if not rows:
        raise ValueError("rows must not be empty")

    actual = np.asarray([float(row["measured_gain"]) for row in rows], dtype=np.float64)
    pred_mean = np.asarray(predicted_mean, dtype=np.float64)
    pred_std = np.asarray(predicted_std, dtype=np.float64)
    pred_score = pred_mean - float(kappa) * pred_std

    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(_resolve_group_id(row, group_key), []).append(index)

    group_reports: list[dict[str, Any]] = []
    spearmans: list[float] = []
    regrets: list[float] = []
    top1_hits: list[float] = []
    topk_hits: list[float] = []

    for group_id, indices in sorted(grouped.items(), key=lambda item: item[0]):
        actual_group = actual[indices]
        pred_mean_group = pred_mean[indices]
        pred_std_group = pred_std[indices]
        pred_score_group = pred_score[indices]
        candidate_rows = [rows[index] for index in indices]
        candidate_ids = [str(row.get("candidate_id", f"row_{index}")) for index, row in zip(indices, candidate_rows)]

        predicted_best_local = int(np.argmax(pred_score_group))
        actual_best_local = int(np.argmax(actual_group))
        predicted_best_actual_gain = float(actual_group[predicted_best_local])
        actual_best_gain = float(actual_group[actual_best_local])
        regret = float(actual_best_gain - predicted_best_actual_gain)
        k_eff = max(1, min(int(top_k), len(indices)))
        threshold = float(np.sort(actual_group)[::-1][k_eff - 1])
        top1_hit = 1.0 if predicted_best_local == actual_best_local else 0.0
        topk_hit = 1.0 if predicted_best_actual_gain >= threshold - 1e-12 else 0.0
        spearman = _spearman_correlation(actual_group.astype(float).tolist(), pred_score_group.astype(float).tolist())

        if spearman is not None:
            spearmans.append(float(spearman))
        regrets.append(regret)
        top1_hits.append(top1_hit)
        topk_hits.append(topk_hit)

        group_reports.append(
            {
                "group_id": group_id,
                "row_count": len(indices),
                "spearman": spearman,
                "top1_hit": bool(top1_hit),
                "topk": k_eff,
                "topk_hit": bool(topk_hit),
                "regret": regret,
                "predicted_best_candidate_id": candidate_ids[predicted_best_local],
                "predicted_best_predicted_gain": float(pred_mean_group[predicted_best_local]),
                "predicted_best_predicted_std": float(pred_std_group[predicted_best_local]),
                "predicted_best_score": float(pred_score_group[predicted_best_local]),
                "predicted_best_actual_gain": predicted_best_actual_gain,
                "actual_best_candidate_id": candidate_ids[actual_best_local],
                "actual_best_gain": actual_best_gain,
            }
        )

    prediction_rows = [
        {
            "candidate_id": str(row.get("candidate_id", f"row_{index}")),
            "group_id": _resolve_group_id(row, group_key),
            "measured_gain": float(actual[index]),
            "predicted_gain_mean": float(pred_mean[index]),
            "predicted_gain_std": float(pred_std[index]),
            "predicted_score": float(pred_score[index]),
            "prediction_error": float(pred_mean[index] - actual[index]),
        }
        for index, row in enumerate(rows)
    ]

    abs_errors = np.abs(pred_mean - actual)
    squared_errors = (pred_mean - actual) ** 2
    sign_matches = [
        1.0 if _sign(float(pred_mean[index])) == _sign(float(actual[index])) else 0.0
        for index in range(len(rows))
    ]

    return {
        "group_key": group_key,
        "row_count": len(rows),
        "group_count": len(grouped),
        "top_k": int(top_k),
        "kappa": float(kappa),
        "metrics": {
            "mae": float(abs_errors.mean()),
            "rmse": float(math.sqrt(float(squared_errors.mean()))),
            "sign_accuracy": float(np.mean(sign_matches)),
            "mean_group_spearman": float(np.mean(spearmans)) if spearmans else None,
            "median_group_spearman": float(np.median(spearmans)) if spearmans else None,
            "top1_hit_rate": float(np.mean(top1_hits)) if top1_hits else None,
            "topk_hit_rate": float(np.mean(topk_hits)) if topk_hits else None,
            "mean_regret": float(np.mean(regrets)) if regrets else None,
            "max_regret": float(np.max(regrets)) if regrets else None,
        },
        "group_reports": group_reports,
        "predictions": prediction_rows,
    }


def cross_validate_surrogate(
    rows: list[dict[str, Any]],
    *,
    model_name: str = "linear",
    bootstrap_models: int = 1,
    group_key: str = "baseline_seed",
    top_k: int = 3,
    kappa: float = 0.0,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(_resolve_group_id(row, group_key), []).append((index, row))

    group_ids = sorted(grouped.keys())
    holdout_predictions: list[dict[str, Any]] = []
    mode = "leave_one_group_out"

    if len(group_ids) <= 1:
        mode = "in_sample_single_group"
        surrogate = build_surrogate(model_name, bootstrap_models=bootstrap_models).fit(rows)
        evaluation = surrogate.evaluate(rows, group_key=group_key, top_k=top_k, kappa=kappa)
        return {
            "mode": mode,
            "model_name": model_name,
            "bootstrap_models": int(bootstrap_models),
            **evaluation,
        }

    for holdout_group in group_ids:
        train_rows = [row for group_id in group_ids if group_id != holdout_group for _, row in grouped[group_id]]
        test_pairs = list(grouped[holdout_group])
        test_rows = [row for _, row in test_pairs]
        surrogate = build_surrogate(model_name, bootstrap_models=bootstrap_models).fit(train_rows)
        predicted_mean = surrogate.predict_mean(test_rows)
        predicted_std = surrogate.predict_std(test_rows)
        for (row_index, row), mean_value, std_value in zip(test_pairs, predicted_mean, predicted_std):
            holdout_predictions.append(
                {
                    "row_index": row_index,
                    "row": row,
                    "predicted_gain_mean": float(mean_value),
                    "predicted_gain_std": float(std_value),
                }
            )

    holdout_predictions.sort(key=lambda item: int(item["row_index"]))

    return {
        "mode": mode,
        "model_name": model_name,
        "bootstrap_models": int(bootstrap_models),
        **evaluate_surrogate_predictions(
            rows,
            predicted_mean=[item["predicted_gain_mean"] for item in holdout_predictions],
            predicted_std=[item["predicted_gain_std"] for item in holdout_predictions],
            group_key=group_key,
            top_k=top_k,
            kappa=kappa,
        ),
    }
