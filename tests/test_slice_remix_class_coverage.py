import json
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_remix.class_coverage import (
    load_class_presence_matrix,
    resolve_annotation_path,
    select_focus_class_spec,
)


class SliceRemixClassCoverageTests(unittest.TestCase):
    def test_resolve_annotation_path_handles_dataset_root_and_annotation_root(self):
        sample_id = "images/train2017/000000000001.jpg"

        from_dataset_root = resolve_annotation_path(sample_id, "/tmp/coco_stuff164k")
        from_ann_root = resolve_annotation_path(sample_id, "/tmp/coco_stuff164k/annotations/train2017")

        self.assertEqual(
            from_dataset_root,
            "/tmp/coco_stuff164k/annotations/train2017/000000000001_labelTrainIds.png",
        )
        self.assertEqual(
            from_ann_root,
            "/tmp/coco_stuff164k/annotations/train2017/000000000001_labelTrainIds.png",
        )

    def test_load_class_presence_matrix_reads_binary_class_presence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ann_root = os.path.join(tmpdir, "annotations", "train2017")
            os.makedirs(ann_root, exist_ok=True)

            mask1 = np.asarray([[0, 1], [255, 1]], dtype=np.uint8)
            mask2 = np.asarray([[2, 2], [255, 2]], dtype=np.uint8)
            Image.fromarray(mask1).save(os.path.join(ann_root, "0001_labelTrainIds.png"))
            Image.fromarray(mask2).save(os.path.join(ann_root, "0002_labelTrainIds.png"))

            matrix = load_class_presence_matrix(
                sample_ids=["images/train2017/0001.jpg", "images/train2017/0002.jpg"],
                annotation_root=tmpdir,
                num_classes=3,
            )

            np.testing.assert_array_equal(
                matrix,
                np.asarray(
                    [
                        [1, 1, 0],
                        [0, 0, 1],
                    ],
                    dtype=np.uint8,
                ),
            )

    def test_select_focus_class_spec_returns_top_gap_classes_in_dataset_order(self):
        baseline = {
            "coco_stuff": {
                "per_class": {
                    "apple": {"IoU": 0.0},
                    "banana": {"IoU": 10.0},
                    "carrot": {"IoU": 1.0},
                    "donut": {"IoU": 0.0},
                }
            }
        }
        full = {
            "coco_stuff": {
                "per_class": {
                    "apple": {"IoU": 12.0},
                    "banana": {"IoU": 14.0},
                    "carrot": {"IoU": 18.0},
                    "donut": {"IoU": 25.0},
                }
            }
        }

        spec = select_focus_class_spec(
            baseline_result=baseline,
            full_result=full,
            min_iou_gap=10.0,
            top_k=2,
        )

        self.assertEqual(spec["class_names"], ["donut", "carrot"])
        self.assertEqual(spec["class_indices"], [3, 2])
        self.assertEqual(len(spec["class_weights"]), 2)


if __name__ == "__main__":
    unittest.main()
