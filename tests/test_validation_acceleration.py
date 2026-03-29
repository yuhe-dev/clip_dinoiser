import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.validation_acceleration import (
    build_validation_payload,
    resolve_proxy_test_cfg,
    sample_dataset_basenames,
    select_full_eval_shortlist,
    subset_dataset_by_basenames,
)


class _DatasetWithInfos:
    def __init__(self, infos, key="data_infos"):
        setattr(self, key, infos)


class ValidationAccelerationTests(unittest.TestCase):
    def test_subset_dataset_by_basenames_filters_data_infos(self):
        dataset = _DatasetWithInfos(
            [
                {"filename": "images/val2017/a.jpg"},
                {"filename": "images/val2017/b.jpg"},
                {"filename": "images/val2017/c.jpg"},
            ]
        )

        subset_dataset_by_basenames(dataset, {"a.jpg", "c.jpg"})

        self.assertEqual(
            [entry["filename"] for entry in dataset.data_infos],
            ["images/val2017/a.jpg", "images/val2017/c.jpg"],
        )

    def test_sample_dataset_basenames_is_deterministic(self):
        dataset = _DatasetWithInfos(
            [{"filename": f"images/val2017/{name}.jpg"} for name in ["a", "b", "c", "d", "e"]]
        )

        first = sample_dataset_basenames(dataset, seed=7, limit=3)
        second = sample_dataset_basenames(dataset, seed=7, limit=3)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_resolve_proxy_test_cfg_whole_and_coarse_slide(self):
        self.assertEqual(resolve_proxy_test_cfg("whole"), {"mode": "whole"})
        self.assertEqual(
            resolve_proxy_test_cfg("coarse_slide"),
            {"mode": "slide", "stride": (448, 448), "crop_size": (448, 448)},
        )

    def test_build_validation_payload_uses_summary_only_for_proxy(self):
        eval_results = {"mIoU": 0.42, "mAcc": 0.51, "aAcc": 0.77, "IoU.cat": 0.1}

        payload = build_validation_payload(
            eval_results=eval_results,
            classes=["cat"],
            validation_mode="proxy",
            used_inference_mode="whole",
        )

        self.assertEqual(payload["validation_mode"], "proxy")
        self.assertEqual(payload["summary"]["mIoU"], 42.0)
        self.assertEqual(payload["proxy_summary"]["mIoU"], 42.0)
        self.assertIsNone(payload["full_summary"])
        self.assertNotIn("per_class", payload)
        self.assertEqual(payload["used_inference_mode"], "whole")

    def test_build_validation_payload_keeps_per_class_for_full(self):
        eval_results = {
            "mIoU": 0.42,
            "mAcc": 0.51,
            "aAcc": 0.77,
            "IoU.cat": 0.1,
            "Acc.cat": 0.2,
        }

        payload = build_validation_payload(
            eval_results=eval_results,
            classes=["cat"],
            validation_mode="full",
            used_inference_mode="slide",
        )

        self.assertEqual(payload["validation_mode"], "full")
        self.assertEqual(payload["summary"]["mIoU"], 42.0)
        self.assertEqual(payload["full_summary"]["mIoU"], 42.0)
        self.assertIsNone(payload["proxy_summary"])
        self.assertEqual(payload["per_class"]["cat"]["IoU"], 10.0)
        self.assertEqual(payload["per_class"]["cat"]["Acc"], 20.0)

    def test_select_full_eval_shortlist_prefers_top_diverse_and_multistep(self):
        candidates = [
            {
                "candidate_id": "top1",
                "proxy_summary": {"mIoU": 40.0},
                "target_mixture": [0.7, 0.2, 0.1],
                "plan_length": 2,
            },
            {
                "candidate_id": "nearby",
                "proxy_summary": {"mIoU": 39.5},
                "target_mixture": [0.69, 0.21, 0.1],
                "plan_length": 2,
            },
            {
                "candidate_id": "diverse",
                "proxy_summary": {"mIoU": 39.0},
                "target_mixture": [0.2, 0.3, 0.5],
                "plan_length": 2,
            },
            {
                "candidate_id": "multistep_best",
                "proxy_summary": {"mIoU": 38.0},
                "target_mixture": [0.25, 0.25, 0.5],
                "plan_length": 4,
            },
            {
                "candidate_id": "multistep_second",
                "proxy_summary": {"mIoU": 37.0},
                "target_mixture": [0.1, 0.4, 0.5],
                "plan_length": 3,
            },
        ]

        shortlist = select_full_eval_shortlist(candidates, top_k=3)

        self.assertEqual(shortlist, ["top1", "diverse", "multistep_best"])

    def test_select_full_eval_shortlist_uses_second_multistep_when_top1_is_multistep(self):
        candidates = [
            {
                "candidate_id": "top1_multistep",
                "proxy_summary": {"mIoU": 41.0},
                "target_mixture": [0.7, 0.2, 0.1],
                "plan_length": 3,
            },
            {
                "candidate_id": "diverse",
                "proxy_summary": {"mIoU": 39.0},
                "target_mixture": [0.2, 0.3, 0.5],
                "plan_length": 2,
            },
            {
                "candidate_id": "multistep_second",
                "proxy_summary": {"mIoU": 38.0},
                "target_mixture": [0.1, 0.4, 0.5],
                "plan_length": 4,
            },
            {
                "candidate_id": "single_step",
                "proxy_summary": {"mIoU": 37.5},
                "target_mixture": [0.68, 0.2, 0.12],
                "plan_length": 1,
            },
        ]

        shortlist = select_full_eval_shortlist(candidates, top_k=3)

        self.assertEqual(shortlist, ["top1_multistep", "diverse", "multistep_second"])


if __name__ == "__main__":
    unittest.main()
