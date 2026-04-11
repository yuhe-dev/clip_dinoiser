from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train surrogate regressors for global mIoU from random-subset feature vectors.")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prepared-dataset-out")
    parser.add_argument("--target-key", default="global_miou")
    parser.add_argument("--rewrite-splits", action="store_true")
    parser.add_argument("--split-seed", type=int, default=20260331)
    parser.add_argument("--train-count", type=int, default=160)
    parser.add_argument("--val-count", type=int, default=16)
    parser.add_argument("--models", default="ridge,pcr,knn,rbf,mlp")
    parser.add_argument(
        "--select-metric",
        choices=["mae", "spearman", "acc_at_10pct_range", "acc_at_20pct_range"],
        default="mae",
    )
    parser.add_argument("--mlp-hidden-dims", default="128x64,64x64,256x128")
    parser.add_argument("--mlp-epochs", type=int, default=400)
    parser.add_argument("--mlp-patience", type=int, default=40)
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def _progress(message: str) -> None:
    print(f"[train_surrogate_global_miou] {message}", file=sys.stderr, flush=True)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(dict(json.loads(line)))
    return rows


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class DatasetSplitConfig:
    train_count: int = 160
    val_count: int = 16
    split_seed: int = 20260331


def build_target_value(row: dict[str, Any]) -> float | None:
    targets = row.get("targets") or {}
    existing = targets.get("global_miou") if isinstance(targets, dict) else None
    if existing is not None:
        return float(existing)

    metrics = row.get("label_metrics") or {}
    if not isinstance(metrics, dict):
        return None
    full_summary = metrics.get("full_summary") or {}
    if isinstance(full_summary, dict) and full_summary.get("mIoU") is not None:
        return float(full_summary["mIoU"])
    summary = metrics.get("summary") or {}
    if isinstance(summary, dict) and summary.get("mIoU") is not None:
        return float(summary["mIoU"])
    return None


def inject_global_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for row in rows:
        row_copy = dict(row)
        targets = dict(row_copy.get("targets") or {})
        target_value = build_target_value(row_copy)
        if target_value is not None:
            targets["global_miou"] = float(target_value)
        row_copy["targets"] = targets
        updated.append(row_copy)
    return updated


def assign_dataset_splits(rows: list[dict[str, Any]], *, config: DatasetSplitConfig) -> list[dict[str, Any]]:
    if int(config.train_count) <= 0:
        raise ValueError("train_count must be positive")
    if int(config.val_count) <= 0:
        raise ValueError("val_count must be positive")
    if len(rows) < int(config.train_count) + int(config.val_count) + 1:
        raise ValueError("dataset is too small for requested train/val/test split")

    updated = [dict(row) for row in rows]
    order = list(range(len(updated)))
    random.Random(int(config.split_seed)).shuffle(order)
    train_cut = int(config.train_count)
    val_cut = int(config.train_count) + int(config.val_count)
    for rank, idx in enumerate(order):
        if rank < train_cut:
            updated[idx]["split"] = "train"
        elif rank < val_cut:
            updated[idx]["split"] = "val"
        else:
            updated[idx]["split"] = "test"
    return updated


def collect_split_rows(
    rows: list[dict[str, Any]],
    *,
    split_name: str,
    target_key: str,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    split_rows: list[dict[str, Any]] = []
    X_rows: list[list[float]] = []
    y_rows: list[float] = []
    for row in rows:
        if str(row.get("split")) != split_name:
            continue
        targets = row.get("targets") or {}
        value = targets.get(target_key) if isinstance(targets, dict) else None
        feature_payload = row.get("feature_payload") or {}
        vector = feature_payload.get("flat_feature_vector") if isinstance(feature_payload, dict) else None
        if value is None or vector is None:
            continue
        split_rows.append(row)
        X_rows.append([float(item) for item in vector])
        y_rows.append(float(value))
    if not split_rows:
        raise ValueError(f"no rows found for split='{split_name}' target='{target_key}'")
    return split_rows, np.asarray(X_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.float64)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[order[end + 1]] == values[order[start]]:
            end += 1
        avg_rank = 0.5 * (start + end) + 1.0
        ranks[order[start : end + 1]] = avg_rank
        start = end + 1
    return ranks


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if y_true.size < 2:
        return None
    rank_true = _rankdata(np.asarray(y_true, dtype=np.float64))
    rank_pred = _rankdata(np.asarray(y_pred, dtype=np.float64))
    if np.std(rank_true) == 0.0 or np.std(rank_pred) == 0.0:
        return None
    corr = np.corrcoef(rank_true, rank_pred)[0, 1]
    return float(corr)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    tolerance: float | None = None,
    train_range: float | None = None,
    baseline_mae: float | None = None,
) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(math.sqrt(float(np.mean(errors**2))))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = None if denom <= 0.0 else float(1.0 - float(np.sum(errors**2)) / denom)
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman": spearman_correlation(y_true, y_pred),
    }
    if tolerance is not None:
        metrics["acc_within_tolerance"] = float(np.mean(np.abs(errors) <= float(tolerance)))
    if train_range is not None and float(train_range) > 0.0:
        range_value = float(train_range)
        metrics["nmae_by_train_range"] = mae / range_value
        for pct in [5, 10, 20]:
            delta = range_value * (pct / 100.0)
            metrics[f"acc_at_{pct}pct_range"] = float(np.mean(np.abs(errors) <= delta))
    if baseline_mae is not None and float(baseline_mae) > 0.0:
        metrics["skill_vs_baseline_mae"] = 1.0 - mae / float(baseline_mae)
    return metrics


def safe_dot(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Avoid NumPy matmul instability observed on this environment for otherwise finite inputs.
    return np.dot(np.ascontiguousarray(left), np.ascontiguousarray(right))


class Standardizer:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean_ = np.asarray(X.mean(axis=0), dtype=np.float64)
        scale = np.asarray(X.std(axis=0), dtype=np.float64)
        scale[scale < 1e-12] = 1.0
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("standardizer must be fit before transform")
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_


class TargetStandardizer:
    def __init__(self) -> None:
        self.mean_: float | None = None
        self.scale_: float | None = None

    def fit(self, y: np.ndarray) -> "TargetStandardizer":
        values = np.asarray(y, dtype=np.float64)
        self.mean_ = float(values.mean())
        scale = float(values.std())
        self.scale_ = 1.0 if scale < 1e-12 else scale
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("target standardizer must be fit before transform")
        return (np.asarray(y, dtype=np.float64) - self.mean_) / self.scale_

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("target standardizer must be fit before inverse_transform")
        return np.asarray(y, dtype=np.float64) * self.scale_ + self.mean_


class RidgeRegressor:
    def __init__(self, alpha: float = 1e-3):
        self.alpha = float(alpha)
        self.scaler_ = Standardizer()
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        Xs = self.scaler_.fit(X).transform(X)
        ones = np.ones((Xs.shape[0], 1), dtype=np.float64)
        design = np.concatenate([ones, Xs], axis=1)
        gram = safe_dot(design.T, design)
        gram += self.alpha * np.eye(gram.shape[0], dtype=np.float64)
        rhs = safe_dot(design.T, np.asarray(y, dtype=np.float64))
        self.coef_ = np.linalg.solve(gram, rhs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("model must be fit before prediction")
        Xs = self.scaler_.transform(X)
        ones = np.ones((Xs.shape[0], 1), dtype=np.float64)
        design = np.concatenate([ones, Xs], axis=1)
        return safe_dot(design, self.coef_)

    def to_payload(self) -> dict[str, Any]:
        if self.coef_ is None:
            raise ValueError("model must be fit before serialization")
        return {
            "type": "ridge",
            "alpha": self.alpha,
            "coef": self.coef_.astype(float).tolist(),
            "feature_mean": self.scaler_.mean_.astype(float).tolist(),
            "feature_scale": self.scaler_.scale_.astype(float).tolist(),
        }


class KNNRegressor:
    def __init__(self, neighbors: int = 9, weighted: bool = True):
        self.neighbors = int(neighbors)
        self.weighted = bool(weighted)
        self.scaler_ = Standardizer()
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        self.X_train_ = self.scaler_.fit(X).transform(X)
        self.y_train_ = np.asarray(y, dtype=np.float64)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.y_train_ is None:
            raise ValueError("model must be fit before prediction")
        X_eval = self.scaler_.transform(X)
        predictions = []
        k = min(self.neighbors, self.X_train_.shape[0])
        for row in X_eval:
            distances = np.sqrt(np.sum((self.X_train_ - row) ** 2, axis=1))
            order = np.argsort(distances)[:k]
            if self.weighted:
                weights = 1.0 / np.clip(distances[order], 1e-8, None)
                pred = float(np.dot(weights, self.y_train_[order]) / weights.sum())
            else:
                pred = float(np.mean(self.y_train_[order]))
            predictions.append(pred)
        return np.asarray(predictions, dtype=np.float64)

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "knn",
            "neighbors": self.neighbors,
            "weighted": self.weighted,
            "feature_mean": self.scaler_.mean_.astype(float).tolist(),
            "feature_scale": self.scaler_.scale_.astype(float).tolist(),
            "train_rows": int(self.X_train_.shape[0]) if self.X_train_ is not None else 0,
        }


class RBFKernelRidgeRegressor:
    def __init__(self, alpha: float = 1e-2, gamma: float = 1.0):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.scaler_ = Standardizer()
        self.X_train_: np.ndarray | None = None
        self.dual_coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RBFKernelRidgeRegressor":
        Xs = self.scaler_.fit(X).transform(X)
        self.X_train_ = Xs
        gram = self._kernel(Xs, Xs)
        gram += self.alpha * np.eye(gram.shape[0], dtype=np.float64)
        self.dual_coef_ = np.linalg.solve(gram, np.asarray(y, dtype=np.float64))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.dual_coef_ is None:
            raise ValueError("model must be fit before prediction")
        Xs = self.scaler_.transform(X)
        return safe_dot(self._kernel(Xs, self.X_train_), self.dual_coef_)

    def _kernel(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_sq = np.sum(left**2, axis=1, keepdims=True)
        right_sq = np.sum(right**2, axis=1, keepdims=True).T
        sqdist = np.clip(left_sq + right_sq - 2.0 * safe_dot(left, right.T), 0.0, None)
        return np.exp(-self.gamma * sqdist)

    def to_payload(self) -> dict[str, Any]:
        if self.X_train_ is None or self.dual_coef_ is None:
            raise ValueError("model must be fit before serialization")
        return {
            "type": "rbf_kernel_ridge",
            "alpha": self.alpha,
            "gamma": self.gamma,
            "feature_mean": self.scaler_.mean_.astype(float).tolist(),
            "feature_scale": self.scaler_.scale_.astype(float).tolist(),
            "train_features": self.X_train_.astype(float).tolist(),
            "dual_coef": self.dual_coef_.astype(float).tolist(),
        }


class PCARidgeRegressor:
    def __init__(self, n_components: int = 16, alpha: float = 1e-2):
        self.n_components = int(n_components)
        self.alpha = float(alpha)
        self.input_scaler_ = Standardizer()
        self.components_: np.ndarray | None = None
        self.ridge_ = RidgeRegressor(alpha=self.alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PCARidgeRegressor":
        Xs = self.input_scaler_.fit(X).transform(X)
        _, _, vt = np.linalg.svd(Xs, full_matrices=False)
        rank = min(self.n_components, vt.shape[0])
        self.components_ = np.asarray(vt[:rank].T, dtype=np.float64)
        projected = safe_dot(Xs, self.components_)
        self.ridge_.fit(projected, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise ValueError("model must be fit before prediction")
        Xs = self.input_scaler_.transform(X)
        projected = safe_dot(Xs, self.components_)
        return self.ridge_.predict(projected)

    def to_payload(self) -> dict[str, Any]:
        if self.components_ is None:
            raise ValueError("model must be fit before serialization")
        return {
            "type": "pca_ridge",
            "n_components": self.n_components,
            "alpha": self.alpha,
            "input_feature_mean": self.input_scaler_.mean_.astype(float).tolist(),
            "input_feature_scale": self.input_scaler_.scale_.astype(float).tolist(),
            "components": self.components_.astype(float).tolist(),
            "ridge": self.ridge_.to_payload(),
        }


class TorchMLPRegressor:
    def __init__(
        self,
        *,
        hidden_dims: tuple[int, ...] = (128, 64),
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 400,
        patience: int = 40,
        batch_size: int = 32,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.device = str(device)
        self.scaler_ = Standardizer()
        self.target_scaler_ = TargetStandardizer()
        self.state_dict_: dict[str, Any] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, *, X_val: np.ndarray, y_val: np.ndarray) -> "TorchMLPRegressor":
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        X_train = self.scaler_.fit(X).transform(X)
        X_eval = self.scaler_.transform(X_val)
        y_train_scaled = self.target_scaler_.fit(y).transform(y)
        y_val_scaled = self.target_scaler_.transform(y_val)

        device = self._resolve_device(torch)
        model = self._build_network(torch, X_train.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        train_inputs = torch.tensor(X_train, dtype=torch.float32, device=device)
        train_targets = torch.tensor(np.asarray(y_train_scaled, dtype=np.float32).reshape(-1, 1), dtype=torch.float32, device=device)
        val_inputs = torch.tensor(X_eval, dtype=torch.float32, device=device)
        val_targets = torch.tensor(np.asarray(y_val_scaled, dtype=np.float32).reshape(-1, 1), dtype=torch.float32, device=device)

        order = np.arange(X_train.shape[0], dtype=np.int64)
        best_metric = math.inf
        bad_epochs = 0
        best_state: dict[str, Any] | None = None
        batch_size = max(1, min(self.batch_size, X_train.shape[0]))

        for epoch in range(self.epochs):
            np.random.default_rng(self.seed + epoch).shuffle(order)
            model.train()
            for start in range(0, len(order), batch_size):
                batch_index = order[start : start + batch_size]
                batch_x = train_inputs[batch_index]
                batch_y = train_targets[batch_index]
                optimizer.zero_grad(set_to_none=True)
                preds = model(batch_x)
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_preds = model(val_inputs)
                val_loss = float(loss_fn(val_preds, val_targets).item())
            if val_loss + 1e-8 < best_metric:
                best_metric = val_loss
                bad_epochs = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is None:
            raise RuntimeError("MLP training did not produce a valid state")
        self.state_dict_ = best_state
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.state_dict_ is None:
            raise ValueError("model must be fit before prediction")
        X_eval = self.scaler_.transform(X)
        device = self._resolve_device(torch)
        model = self._build_network(torch, X_eval.shape[1]).to(device)
        model.load_state_dict(self.state_dict_)
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X_eval, dtype=torch.float32, device=device)
            preds = model(inputs).detach().cpu().numpy().reshape(-1)
        return self.target_scaler_.inverse_transform(np.asarray(preds, dtype=np.float64))

    def save_artifact(self, path: str) -> str:
        import torch

        if self.state_dict_ is None:
            raise ValueError("model must be fit before serialization")
        output_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(
            {
                "type": "torch_mlp",
                "hidden_dims": list(self.hidden_dims),
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "epochs": self.epochs,
                "patience": self.patience,
                "batch_size": self.batch_size,
                "seed": self.seed,
                "feature_mean": self.scaler_.mean_.astype(float).tolist(),
                "feature_scale": self.scaler_.scale_.astype(float).tolist(),
                "target_mean": self.target_scaler_.mean_,
                "target_scale": self.target_scaler_.scale_,
                "state_dict": self.state_dict_,
            },
            output_path,
        )
        return output_path

    def _resolve_device(self, torch_module) -> str:
        if self.device == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch_module.cuda.is_available():
            return "cpu"
        return self.device

    def _build_network(self, torch_module, input_dim: int):
        from torch import nn

        dims = [input_dim, *self.hidden_dims, 1]
        layers: list[Any] = []
        for left, right in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(left, right))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        return nn.Sequential(*layers)


def _maybe_rewrite_splits(rows: list[dict[str, Any]], args: argparse.Namespace, log_fn=_progress) -> list[dict[str, Any]]:
    split_names = {str(row.get("split")) for row in rows}
    needs_rewrite = bool(args.rewrite_splits) or split_names == {"train"}
    if not needs_rewrite:
        return rows
    log_fn(
        f"rewriting split assignments train={int(args.train_count)} val={int(args.val_count)} "
        f"test={len(rows) - int(args.train_count) - int(args.val_count)} seed={int(args.split_seed)}"
    )
    return assign_dataset_splits(
        rows,
        config=DatasetSplitConfig(
            train_count=int(args.train_count),
            val_count=int(args.val_count),
            split_seed=int(args.split_seed),
        ),
    )


def _split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("split"))] = counts.get(str(row.get("split")), 0) + 1
    return counts


def _build_model_families(args: argparse.Namespace, X_train: np.ndarray) -> dict[str, list[tuple[str, Any]]]:
    models: dict[str, list[tuple[str, Any]]] = {}
    requested = [token.strip() for token in str(args.models).split(",") if token.strip()]
    for name in requested:
        if name == "ridge":
            models[name] = [(f"alpha={alpha:g}", RidgeRegressor(alpha=alpha)) for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]]
        elif name == "pcr":
            models[name] = [
                (f"k={k},alpha={alpha:g}", PCARidgeRegressor(n_components=k, alpha=alpha))
                for k in [4, 8, 16, 32, 64, 96]
                for alpha in [1e-3, 1e-2, 1e-1]
            ]
        elif name == "knn":
            models[name] = [
                (f"k={k},weighted={weighted}", KNNRegressor(neighbors=k, weighted=weighted))
                for k in [3, 5, 9, 15, 31]
                for weighted in [True, False]
            ]
        elif name == "rbf":
            sample = min(64, X_train.shape[0])
            scaler = Standardizer().fit(X_train)
            sample_X = scaler.transform(X_train[:sample])
            sqdist = np.sum((sample_X[:, None, :] - sample_X[None, :, :]) ** 2, axis=-1)
            base_scale = float(np.median(sqdist[sqdist > 0])) if np.any(sqdist > 0) else 1.0
            base_gamma = 1.0 / max(base_scale, 1e-6)
            models[name] = [
                (f"alpha={alpha:g},gamma={base_gamma * scale:.5f}", RBFKernelRidgeRegressor(alpha=alpha, gamma=base_gamma * scale))
                for alpha in [1e-3, 1e-2, 1e-1, 1.0]
                for scale in [0.25, 1.0, 4.0]
            ]
        elif name == "mlp":
            hidden_specs = []
            for token in str(args.mlp_hidden_dims).split(","):
                item = token.strip()
                if not item:
                    continue
                hidden_specs.append(tuple(int(part) for part in item.split("x") if part.strip()))
            if not hidden_specs:
                hidden_specs = [(128, 64)]
            models[name] = [
                (
                    f"hidden={'x'.join(str(v) for v in hidden_dims)}",
                    TorchMLPRegressor(
                        hidden_dims=hidden_dims,
                        learning_rate=float(args.mlp_learning_rate),
                        weight_decay=float(args.mlp_weight_decay),
                        epochs=int(args.mlp_epochs),
                        patience=int(args.mlp_patience),
                        batch_size=int(args.mlp_batch_size),
                        seed=0,
                        device=str(args.device),
                    ),
                )
                for hidden_dims in hidden_specs
            ]
        else:
            raise ValueError(f"unsupported model family '{name}'")
    return models


def _select_better_candidate(
    best: dict[str, Any] | None,
    candidate: dict[str, Any],
    *,
    select_metric: str = "mae",
) -> bool:
    if best is None:
        return True
    best_metrics = best["val_metrics"]
    cand_metrics = candidate["val_metrics"]
    best_mae = float(best_metrics["mae"])
    cand_mae = float(cand_metrics["mae"])
    best_sp = best_metrics.get("spearman")
    cand_sp = cand_metrics.get("spearman")
    best_sp = -math.inf if best_sp is None else float(best_sp)
    cand_sp = -math.inf if cand_sp is None else float(cand_sp)

    if select_metric == "mae":
        if cand_mae + 1e-12 < best_mae:
            return True
        if abs(cand_mae - best_mae) <= 1e-12:
            return cand_sp > best_sp
        return False

    if select_metric == "spearman":
        if cand_sp > best_sp + 1e-12:
            return True
        if abs(cand_sp - best_sp) <= 1e-12:
            return cand_mae < best_mae - 1e-12
        return False

    best_metric = best_metrics.get(select_metric)
    cand_metric = cand_metrics.get(select_metric)
    best_metric = -math.inf if best_metric is None else float(best_metric)
    cand_metric = -math.inf if cand_metric is None else float(cand_metric)
    if cand_metric > best_metric + 1e-12:
        return True
    if abs(cand_metric - best_metric) <= 1e-12:
        if cand_mae + 1e-12 < best_mae:
            return True
        if abs(cand_mae - best_mae) <= 1e-12:
            return cand_sp > best_sp
    return False


def _save_predictions(path: str, rows: list[dict[str, Any]], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    payload = []
    for row, actual, predicted in zip(rows, y_true.tolist(), y_pred.tolist()):
        payload.append(
            {
                "experiment_id": row["experiment_id"],
                "split": row["split"],
                "actual": float(actual),
                "predicted": float(predicted),
                "error": float(predicted - actual),
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run(args: argparse.Namespace, log_fn=_progress) -> int:
    rows = read_jsonl(os.path.abspath(args.dataset_jsonl))
    if not rows:
        raise ValueError("dataset rows must not be empty")
    rows = [row for row in rows if row.get("label_ready")]
    if not rows:
        raise ValueError("no label-ready rows available")
    rows = inject_global_targets(rows)
    rows = _maybe_rewrite_splits(rows, args, log_fn=log_fn)

    prepared_out = (
        os.path.abspath(args.prepared_dataset_out)
        if args.prepared_dataset_out
        else os.path.join(os.path.abspath(args.output_dir), "prepared_surrogate_global_miou.jsonl")
    )
    os.makedirs(os.path.abspath(args.output_dir), exist_ok=True)
    write_jsonl(prepared_out, rows)
    log_fn(f"wrote prepared dataset path={prepared_out} splits={_split_counts(rows)}")

    train_rows, X_train, y_train = collect_split_rows(rows, split_name="train", target_key=str(args.target_key))
    val_rows, X_val, y_val = collect_split_rows(rows, split_name="val", target_key=str(args.target_key))
    test_rows, X_test, y_test = collect_split_rows(rows, split_name="test", target_key=str(args.target_key))
    train_range = float(y_train.max() - y_train.min())
    train_mean = float(y_train.mean())
    val_baseline_mae = float(np.mean(np.abs(y_val - train_mean)))
    test_baseline_mae = float(np.mean(np.abs(y_test - train_mean)))
    log_fn(
        f"loaded matrices target={args.target_key} train={X_train.shape} val={X_val.shape} test={X_test.shape}"
    )

    families = _build_model_families(args, X_train)
    results: dict[str, Any] = {
        "dataset_jsonl": os.path.abspath(args.dataset_jsonl),
        "prepared_dataset": prepared_out,
        "target_key": str(args.target_key),
        "selection_metric": str(args.select_metric),
        "splits": _split_counts(rows),
        "train_target_range": train_range,
        "tolerance_deltas": {
            "5pct_range": train_range * 0.05,
            "10pct_range": train_range * 0.10,
            "20pct_range": train_range * 0.20,
        },
        "models": {},
    }

    for family_name, candidates in families.items():
        best_result: dict[str, Any] | None = None
        family_candidates: list[dict[str, Any]] = []
        family_dir = os.path.join(os.path.abspath(args.output_dir), family_name)
        os.makedirs(family_dir, exist_ok=True)
        log_fn(f"training family={family_name} candidates={len(candidates)}")
        for candidate_name, model in candidates:
            if isinstance(model, TorchMLPRegressor):
                model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            else:
                model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            candidate_result = {
                "candidate_name": candidate_name,
                "val_metrics": regression_metrics(
                    y_val,
                    val_pred,
                    tolerance=train_range * 0.10,
                    train_range=train_range,
                    baseline_mae=val_baseline_mae,
                ),
                "test_metrics": regression_metrics(
                    y_test,
                    test_pred,
                    tolerance=train_range * 0.10,
                    train_range=train_range,
                    baseline_mae=test_baseline_mae,
                ),
                "model": model,
                "val_pred": val_pred,
                "test_pred": test_pred,
            }
            family_candidates.append(
                {
                    "candidate_name": candidate_name,
                    "val_metrics": candidate_result["val_metrics"],
                    "test_metrics": candidate_result["test_metrics"],
                }
            )
            if _select_better_candidate(best_result, candidate_result, select_metric=str(args.select_metric)):
                best_result = candidate_result

        if best_result is None:
            raise RuntimeError(f"no candidate trained for family='{family_name}'")

        best_model = best_result.pop("model")
        model_artifact_path = None
        if isinstance(best_model, TorchMLPRegressor):
            model_artifact_path = best_model.save_artifact(os.path.join(family_dir, "best_model.pt"))
        else:
            model_artifact_path = os.path.join(family_dir, "best_model.json")
            with open(model_artifact_path, "w", encoding="utf-8") as f:
                json.dump(best_model.to_payload(), f, indent=2, ensure_ascii=False)

        _save_predictions(os.path.join(family_dir, "val_predictions.json"), val_rows, y_val, best_result["val_pred"])
        _save_predictions(os.path.join(family_dir, "test_predictions.json"), test_rows, y_test, best_result["test_pred"])

        results["models"][family_name] = {
            "candidate_name": best_result["candidate_name"],
            "val_metrics": best_result["val_metrics"],
            "test_metrics": best_result["test_metrics"],
            "model_artifact_path": model_artifact_path,
            "candidates": family_candidates,
        }
        log_fn(
            f"best family={family_name} candidate={best_result['candidate_name']} "
            f"val_{args.select_metric}={best_result['val_metrics'].get(str(args.select_metric))} "
            f"val_mae={best_result['val_metrics']['mae']:.4f}"
        )

    summary_path = os.path.join(os.path.abspath(args.output_dir), "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log_fn(f"wrote training summary path={summary_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
