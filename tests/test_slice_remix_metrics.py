import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_remix.metrics import extract_metric_value, load_experiment_metrics


class SliceRemixMetricsTests(unittest.TestCase):
    def test_extract_metric_value_reads_summary_metric_path(self):
        payload = {
            "coco_stuff": {
                "summary": {"mIoU": 24.76, "mAcc": 41.83, "aAcc": 38.95},
            }
        }

        self.assertEqual(extract_metric_value(payload, "coco_stuff.summary.mIoU"), 24.76)

    def test_load_experiment_metrics_returns_summary_and_per_class(self):
        payload = {
            "coco_stuff": {
                "summary": {"mIoU": 24.76, "mAcc": 41.83, "aAcc": 38.95},
                "per_class": {"person": {"IoU": 12.5, "Acc": 13.2}},
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "result.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            loaded = load_experiment_metrics(path)

        self.assertEqual(loaded["summary"]["mIoU"], 24.76)
        self.assertEqual(loaded["per_class"]["person"]["IoU"], 12.5)


if __name__ == "__main__":
    unittest.main()
