import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.backfill_small_ratio_counts import backfill_small_ratio_counts_for_records


class TestBackfillSmallRatioCounts(unittest.TestCase):
    def test_backfill_populates_real_connected_component_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subset_root = os.path.join(tmpdir, "subset")
            annotation_dir = os.path.join(subset_root, "annotations", "train2017")
            os.makedirs(annotation_dir, exist_ok=True)

            mask = np.full((10, 10), 255, dtype=np.uint8)
            mask[0, 0] = 0
            mask[2:4, 2:4] = 1
            mask[5:10, 5:10] = 2
            annotation_rel = "annotations/train2017/0001_labelTrainIds.png"
            Image.fromarray(mask).save(os.path.join(subset_root, annotation_rel))

            records = [
                {
                    "image_rel": "images/train2017/0001.jpg",
                    "annotation_rel": annotation_rel,
                    "small_ratio_raw": np.asarray([0.0] * 16, dtype=np.float32),
                }
            ]

            logs = []
            updated = backfill_small_ratio_counts_for_records(
                records=records,
                subset_root=subset_root,
                thresholds=np.geomspace(0.001, 0.2, 16).tolist(),
                ignore_index=255,
                use_things_only=False,
                progress_interval=1,
                log_fn=logs.append,
            )

            self.assertEqual(len(updated), 1)
            self.assertEqual(updated[0]["small_ratio_num_values"], 3)
            self.assertTrue(any("processed 1/1" in msg for msg in logs))


if __name__ == "__main__":
    unittest.main()
