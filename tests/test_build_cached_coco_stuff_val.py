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


from clip_dinoiser.build_cached_coco_stuff_val import build_cache, resolve_mask_name
from clip_dinoiser.slice_remix.eval_cache import load_cache_manifest


class BuildCachedCocoStuffValTests(unittest.TestCase):
    def test_build_cache_writes_manifest_meta_and_npy_payloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = os.path.join(tmpdir, "coco_stuff164k")
            image_dir = os.path.join(data_root, "images", "val2017")
            mask_dir = os.path.join(data_root, "annotations", "val2017")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            image_name = "000000000123.jpg"
            image = np.full((640, 480, 3), 127, dtype=np.uint8)
            mask = np.full((640, 480), 5, dtype=np.uint8)
            Image.fromarray(image).save(os.path.join(image_dir, image_name))
            Image.fromarray(mask).save(os.path.join(mask_dir, resolve_mask_name(image_name)))

            output_root = os.path.join(tmpdir, "cache")
            written = build_cache(
                data_root=data_root,
                output_root=output_root,
                img_scale=(2048, 448),
                limit=0,
            )

            self.assertEqual(written, 1)
            manifest = load_cache_manifest(os.path.join(output_root, "manifest.jsonl"))
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["basename"], image_name)

            cached_image = np.load(os.path.join(output_root, manifest[0]["image_npy"]))
            cached_mask = np.load(os.path.join(output_root, manifest[0]["mask_npy"]))
            self.assertEqual(tuple(cached_image.shape), (597, 448, 3))
            self.assertEqual(tuple(cached_mask.shape), (640, 480))

            with open(os.path.join(output_root, "meta.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["count"], 1)
            self.assertEqual(meta["img_scale"], [2048, 448])
            self.assertTrue(meta["keep_ratio"])


if __name__ == "__main__":
    unittest.main()
