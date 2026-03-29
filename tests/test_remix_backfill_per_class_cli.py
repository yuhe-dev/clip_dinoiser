import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_remix_backfill_per_class import main


class RemixBackfillPerClassCliTests(unittest.TestCase):
    def test_backfill_writes_per_class_into_existing_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "cand_0")
            os.makedirs(run_dir, exist_ok=True)
            result_path = os.path.join(run_dir, "result.json")
            log_path = os.path.join(run_dir, "log.txt")

            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "validation_mode": "proxy",
                        "validation_config": {},
                        "coco_stuff": {
                            "validation_mode": "proxy",
                            "summary": {"mIoU": 20.0, "mAcc": 30.0, "aAcc": 40.0},
                        },
                    },
                    f,
                    indent=2,
                )

            with open(log_path, "w", encoding="utf-8") as f:
                f.write(
                    """
[t] INFO per class results:
[t] INFO 
+------------------+-------+-------+
|      Class       |  IoU  |  Acc  |
+------------------+-------+-------+
|      person      | 16.26 | 17.36 |
|     bicycle      | 44.29 | 83.13 |
+------------------+-------+-------+
[t] INFO Summary:
"""
                )

            exit_code = main(["--runs-root", tmpdir])
            self.assertEqual(exit_code, 0)

            with open(result_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["coco_stuff"]["per_class"]["person"]["IoU"], 16.26)
            self.assertEqual(payload["coco_stuff"]["per_class"]["bicycle"]["Acc"], 83.13)


if __name__ == "__main__":
    unittest.main()
