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


from clip_dinoiser.tools.sample_voc20_subset import build_split_records
from clip_dinoiser.tools.sample_coco_stuff_subset import materialize_subset, sample_records, write_metadata_files


class Voc20SubsetSamplingTests(unittest.TestCase):
    def test_build_split_records_reads_from_split_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "JPEGImages").mkdir(parents=True)
            (root / "SegmentationClass").mkdir(parents=True)
            (root / "ImageSets" / "Segmentation").mkdir(parents=True)
            (root / "JPEGImages" / "2007_000001.jpg").write_bytes(b"img-1")
            (root / "JPEGImages" / "2007_000002.jpg").write_bytes(b"img-2")
            (root / "SegmentationClass" / "2007_000001.png").write_bytes(b"ann-1")
            (root / "SegmentationClass" / "2007_000002.png").write_bytes(b"ann-2")
            split_path = root / "ImageSets" / "Segmentation" / "train.txt"
            split_path.write_text("2007_000001\n2007_000002\n", encoding="utf-8")

            records = build_split_records(root, split_path)

            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["image_rel"], "JPEGImages/2007_000001.jpg")
            self.assertEqual(records[0]["annotation_rel"], "SegmentationClass/2007_000001.png")

    def test_voc20_subset_helpers_materialize_and_write_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "VOC2012"
            (source_root / "JPEGImages").mkdir(parents=True)
            (source_root / "SegmentationClass").mkdir(parents=True)
            (source_root / "ImageSets" / "Segmentation").mkdir(parents=True)
            for stem in ("2007_000001", "2007_000002"):
                (source_root / "JPEGImages" / f"{stem}.jpg").write_bytes(f"img-{stem}".encode("utf-8"))
                (source_root / "SegmentationClass" / f"{stem}.png").write_bytes(f"ann-{stem}".encode("utf-8"))
            split_path = source_root / "ImageSets" / "Segmentation" / "train.txt"
            split_path.write_text("2007_000001\n2007_000002\n", encoding="utf-8")

            records = build_split_records(source_root, split_path)
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
                index_path, config_path = write_metadata_files(
                    subset_root=subset_root,
                    source_root=source_root,
                    sampled_records=sampled,
                    seed=0,
                    sample_size=2,
                    split="ImageSets/Segmentation/train.txt",
                    link_mode="copy",
                    dataset_name="voc20_subset",
                    verbose=True,
                )

            stdout = output.getvalue()
            self.assertIn("Materializing 2 samples", stdout)
            self.assertTrue((subset_root / "JPEGImages" / "2007_000001.jpg").exists())
            self.assertTrue((subset_root / "SegmentationClass" / "2007_000001.png").exists())

            loaded = np.load(index_path, allow_pickle=True)
            self.assertEqual(loaded[0]["image_rel"], "JPEGImages/2007_000001.jpg")
            self.assertEqual(loaded[0]["annotation_rel"], "SegmentationClass/2007_000001.png")

            config = json.loads(config_path.read_text())
            self.assertEqual(config["dataset_name"], "voc20_subset")


if __name__ == "__main__":
    unittest.main()
