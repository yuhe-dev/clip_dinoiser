import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_remix.eval_cache import (
    build_cache_record,
    load_cache_manifest,
    resolve_keep_ratio_size,
)


class EvalCacheTests(unittest.TestCase):
    def test_resolve_keep_ratio_size_respects_mmseg_scale_constraints(self):
        self.assertEqual(resolve_keep_ratio_size(height=640, width=480, scale=(2048, 448)), (597, 448))
        self.assertEqual(resolve_keep_ratio_size(height=1024, width=2048, scale=(2048, 448)), (448, 896))

    def test_build_cache_record_stores_paths_and_shapes(self):
        record = build_cache_record(
            basename="000000000123.jpg",
            image_rel_path="images/000000000123.npy",
            mask_rel_path="masks/000000000123.npy",
            ori_shape=(640, 480, 3),
            cached_img_shape=(597, 448, 3),
            scale_factor=(0.9333333, 0.9328125, 0.9333333, 0.9328125),
        )

        self.assertEqual(record["basename"], "000000000123.jpg")
        self.assertEqual(record["image_npy"], "images/000000000123.npy")
        self.assertEqual(record["mask_npy"], "masks/000000000123.npy")
        self.assertEqual(record["ori_shape"], [640, 480, 3])
        self.assertEqual(record["cached_img_shape"], [597, 448, 3])
        self.assertEqual(len(record["scale_factor"]), 4)

    def test_load_cache_manifest_reads_jsonl_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "manifest.jsonl")
            rows = [
                {"basename": "a.jpg", "image_npy": "images/a.npy", "mask_npy": "masks/a.npy"},
                {"basename": "b.jpg", "image_npy": "images/b.npy", "mask_npy": "masks/b.npy"},
            ]
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            loaded = load_cache_manifest(path)
            self.assertEqual(loaded, rows)


if __name__ == "__main__":
    unittest.main()
