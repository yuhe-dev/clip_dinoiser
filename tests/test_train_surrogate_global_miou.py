import json
import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.train_surrogate_global_miou import (
    DatasetSplitConfig,
    PCARidgeRegressor,
    RidgeRegressor,
    _select_better_candidate,
    assign_dataset_splits,
    build_target_value,
    collect_split_rows,
    regression_metrics,
    safe_dot,
)


class TrainSurrogateGlobalMiouTests(unittest.TestCase):
    def test_build_target_value_prefers_full_summary_miou(self):
        row = {
            "label_ready": True,
            "label_metrics": {
                "full_summary": {"mIoU": 24.33},
                "summary": {"mIoU": 24.11},
            },
        }
        self.assertAlmostEqual(build_target_value(row), 24.33)

    def test_assign_dataset_splits_rewrites_to_requested_counts(self):
        rows = [{"experiment_id": f"exp_{idx:03d}", "split": "train"} for idx in range(192)]
        updated = assign_dataset_splits(
            rows,
            config=DatasetSplitConfig(train_count=160, val_count=16, split_seed=20260331),
        )
        counts = {}
        for row in updated:
            counts[row["split"]] = counts.get(row["split"], 0) + 1
        self.assertEqual(counts, {"train": 160, "val": 16, "test": 16})

    def test_collect_split_rows_builds_feature_matrix_and_targets(self):
        rows = [
            {
                "experiment_id": "exp_a",
                "split": "train",
                "feature_payload": {"flat_feature_vector": [1.0, 2.0, 3.0]},
                "targets": {"global_miou": 24.1},
            },
            {
                "experiment_id": "exp_b",
                "split": "val",
                "feature_payload": {"flat_feature_vector": [4.0, 5.0, 6.0]},
                "targets": {"global_miou": 24.2},
            },
        ]
        split_rows, X, y = collect_split_rows(rows, split_name="train", target_key="global_miou")
        self.assertEqual([row["experiment_id"] for row in split_rows], ["exp_a"])
        np.testing.assert_allclose(X, np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64))
        np.testing.assert_allclose(y, np.asarray([24.1], dtype=np.float64))

    def test_ridge_regressor_learns_simple_linear_signal(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(64, 5))
        y = 0.5 * X[:, 0] - 1.2 * X[:, 1] + 0.3 * X[:, 2] + 24.0
        model = RidgeRegressor(alpha=1e-3).fit(X, y)
        preds = model.predict(X)
        mae = float(np.mean(np.abs(preds - y)))
        self.assertLess(mae, 0.05)

    def test_pca_ridge_regressor_learns_low_rank_signal(self):
        rng = np.random.default_rng(11)
        latent = rng.normal(size=(96, 4))
        projection = rng.normal(size=(4, 20))
        X = safe_dot(latent, projection)
        weights = rng.normal(size=20)
        y = safe_dot(X, weights) + 24.0
        model = PCARidgeRegressor(n_components=6, alpha=1e-2).fit(X, y)
        preds = model.predict(X)
        mae = float(np.mean(np.abs(preds - y)))
        self.assertLess(mae, 0.08)

    def test_regression_metrics_reports_tolerance_accuracy(self):
        y_true = np.asarray([24.20, 24.25, 24.30, 24.35], dtype=np.float64)
        y_pred = np.asarray([24.21, 24.24, 24.31, 24.33], dtype=np.float64)
        metrics = regression_metrics(
            y_true,
            y_pred,
            tolerance=0.015,
            train_range=0.20,
            baseline_mae=0.03,
        )
        self.assertAlmostEqual(metrics["acc_within_tolerance"], 0.75)
        self.assertAlmostEqual(metrics["nmae_by_train_range"], metrics["mae"] / 0.20)
        self.assertAlmostEqual(metrics["skill_vs_baseline_mae"], 1.0 - metrics["mae"] / 0.03)

    def test_select_better_candidate_can_optimize_for_tolerance_accuracy(self):
        best = {
            "val_metrics": {
                "mae": 0.010,
                "spearman": 0.20,
                "acc_at_10pct_range": 0.50,
            }
        }
        candidate = {
            "val_metrics": {
                "mae": 0.012,
                "spearman": 0.10,
                "acc_at_10pct_range": 0.625,
            }
        }
        self.assertTrue(_select_better_candidate(best, candidate, select_metric="acc_at_10pct_range"))


if __name__ == "__main__":
    unittest.main()
