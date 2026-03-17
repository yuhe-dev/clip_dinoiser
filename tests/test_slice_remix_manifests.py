import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_remix.manifests import load_subset_manifest


class SliceRemixManifestTests(unittest.TestCase):
    def test_load_subset_manifest_prefers_embedded_sample_paths(self):
        payload = {
            "candidate_id": "cand_0",
            "sample_ids": ["a.jpg", "b.jpg"],
            "sample_paths": ["/tmp/a.jpg", "/tmp/b.jpg"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subset.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            manifest = load_subset_manifest(path)

        self.assertEqual(manifest.sample_ids, ["a.jpg", "b.jpg"])
        self.assertEqual(manifest.sample_paths, ["/tmp/a.jpg", "/tmp/b.jpg"])

    def test_load_subset_manifest_can_resolve_paths_from_pool_root(self):
        payload = {
            "candidate_id": "cand_0",
            "sample_ids": ["images/train2017/a.jpg", "images/train2017/b.jpg"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subset.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            manifest = load_subset_manifest(path, pool_image_root="/data/coco")

        self.assertEqual(
            manifest.sample_paths,
            ["/data/coco/images/train2017/a.jpg", "/data/coco/images/train2017/b.jpg"],
        )


if __name__ == "__main__":
    unittest.main()
