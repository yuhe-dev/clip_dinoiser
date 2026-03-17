import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_remix.labels import attach_measured_gain


class SliceRemixLabelsTests(unittest.TestCase):
    def test_attach_measured_gain_computes_candidate_minus_baseline(self):
        baseline_payload = {"coco_stuff": {"summary": {"mIoU": 24.0}}}
        candidate_payload = {"coco_stuff": {"summary": {"mIoU": 25.5}}}
        row = {"candidate_id": "cand_0"}

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = os.path.join(tmpdir, "baseline.json")
            candidate_path = os.path.join(tmpdir, "candidate.json")
            with open(baseline_path, "w", encoding="utf-8") as f:
                json.dump(baseline_payload, f)
            with open(candidate_path, "w", encoding="utf-8") as f:
                json.dump(candidate_payload, f)

            labeled = attach_measured_gain(
                row,
                baseline_result_path=baseline_path,
                candidate_result_path=candidate_path,
                metric_path="coco_stuff.summary.mIoU",
            )

        self.assertEqual(labeled["baseline_metric_value"], 24.0)
        self.assertEqual(labeled["candidate_metric_value"], 25.5)
        self.assertEqual(labeled["measured_gain"], 1.5)


if __name__ == "__main__":
    unittest.main()
