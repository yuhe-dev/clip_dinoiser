import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.research_harness.noise_floor import summarize_metric_rows


class ResearchHarnessNoiseFloorTests(unittest.TestCase):
    def test_summarize_metric_rows_computes_spread(self):
        rows = [
            {"experiment_id": "a", "source": "random_subset", "budget": 1000, "subset_seed": 0, "training_seed": 0, "label_metrics": {"summary": {"mIoU": 24.22}}},
            {"experiment_id": "b", "source": "random_subset", "budget": 1000, "subset_seed": 1, "training_seed": 0, "label_metrics": {"full_summary": {"mIoU": 24.29}}},
            {"experiment_id": "c", "source": "random_subset", "budget": 1000, "subset_seed": 2, "training_seed": 0, "label_metrics": {"summary": {"mIoU": 24.33}}},
            {"experiment_id": "d", "source": "random_subset", "budget": 1000, "subset_seed": 3, "training_seed": 0, "label_metrics": {"summary": {"mIoU": 24.36}}},
        ]

        summary = summarize_metric_rows(rows, metric_name="mIoU")

        self.assertEqual(summary["count"], 4)
        self.assertAlmostEqual(summary["min"], 24.22)
        self.assertAlmostEqual(summary["max"], 24.36)
        self.assertAlmostEqual(summary["range"], 0.14)
        self.assertAlmostEqual(summary["mean"], 24.30)
        self.assertEqual(summary["budget_counts"], {"1000": 4})
        self.assertEqual(summary["subset_seed_count"], 4)


if __name__ == "__main__":
    unittest.main()
