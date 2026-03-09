import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.tools.sample_coco_stuff_subset import (
    build_pair_records,
    materialize_subset,
    sample_records,
    write_metadata_files,
)


class TestCocoStuffSubsetSampling(unittest.TestCase):
    def test_collect_pairs_matches_images_with_labeltrainids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images" / "train2017"
            ann_dir = root / "annotations" / "train2017"
            images_dir.mkdir(parents=True)
            ann_dir.mkdir(parents=True)

            (images_dir / "0001.jpg").write_bytes(b"img-1")
            (images_dir / "0002.jpg").write_bytes(b"img-2")
            (images_dir / "0003.png").write_bytes(b"img-3")
            (ann_dir / "0001_labelTrainIds.png").write_bytes(b"ann-1")
            (ann_dir / "0003_labelTrainIds.png").write_bytes(b"ann-3")

            records = build_pair_records(images_dir, ann_dir, root)

            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["image_rel"], "images/train2017/0001.jpg")
            self.assertEqual(records[0]["annotation_rel"], "annotations/train2017/0001_labelTrainIds.png")
            self.assertEqual(records[1]["image_rel"], "images/train2017/0003.png")

    def test_sample_pairs_is_deterministic(self):
        records = [{"stem": f"{index:04d}"} for index in range(10)]

        sample_a = sample_records(records, sample_size=5, seed=0)
        sample_b = sample_records(records, sample_size=5, seed=0)
        sample_c = sample_records(records, sample_size=5, seed=1)

        self.assertEqual(sample_a, sample_b)
        self.assertNotEqual(sample_a, sample_c)
        self.assertEqual(len(sample_a), 5)

    def test_write_subset_manifest_preserves_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subset_root = Path(tmpdir) / "subset"
            records = [
                {
                    "stem": "0001",
                    "image_rel": "images/train2017/0001.jpg",
                    "annotation_rel": "annotations/train2017/0001_labelTrainIds.png",
                }
            ]

            index_path, config_path = write_metadata_files(
                subset_root=subset_root,
                source_root=Path("/data/coco_stuff164k"),
                sampled_records=records,
                seed=0,
                sample_size=1,
                split="train2017",
                link_mode="symlink",
            )

            self.assertTrue(index_path.exists())
            self.assertTrue(config_path.exists())

            loaded = np.load(index_path, allow_pickle=True)
            self.assertEqual(loaded.shape[0], 1)
            self.assertEqual(loaded[0]["image_rel"], "images/train2017/0001.jpg")

            config = json.loads(config_path.read_text())
            self.assertEqual(config["seed"], 0)
            self.assertEqual(config["sample_size"], 1)
            self.assertEqual(config["link_mode"], "symlink")

    def test_main_helpers_create_subset_tree_and_logs_examples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "coco_stuff164k"
            images_dir = source_root / "images" / "train2017"
            ann_dir = source_root / "annotations" / "train2017"
            images_dir.mkdir(parents=True)
            ann_dir.mkdir(parents=True)

            for stem in ("0001", "0002"):
                (images_dir / f"{stem}.jpg").write_bytes(f"img-{stem}".encode("utf-8"))
                (ann_dir / f"{stem}_labelTrainIds.png").write_bytes(f"ann-{stem}".encode("utf-8"))

            records = build_pair_records(images_dir, ann_dir, source_root)
            sampled = sample_records(records, sample_size=2, seed=0)
            subset_root = root / "subset"

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                materialize_subset(
                    source_root=source_root,
                    subset_root=subset_root,
                    sampled_records=sampled,
                    link_mode="copy",
                    verbose=True,
                )
                write_metadata_files(
                    subset_root=subset_root,
                    source_root=source_root,
                    sampled_records=sampled,
                    seed=0,
                    sample_size=2,
                    split="train2017",
                    link_mode="copy",
                    verbose=True,
                )

            stdout = output.getvalue()
            self.assertIn("Materializing 2 samples", stdout)
            self.assertIn("sample_index.npy", stdout)
            self.assertIn("Example records", stdout)

            self.assertTrue((subset_root / "images" / "train2017" / "0001.jpg").exists())
            self.assertTrue((subset_root / "annotations" / "train2017" / "0001_labelTrainIds.png").exists())


if __name__ == "__main__":
    unittest.main()
