import os
import unittest

from run_supervised_probe_experiment import build_parser
from research_harness.supervised_probe import build_supervised_probe_cfg


class SupervisedProbeDatasetCfgTest(unittest.TestCase):
    def test_cli_allows_full_train_split_without_manifest(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--dataset",
                "voc20",
                "--output-dir",
                "/tmp/probe-out",
            ]
        )
        self.assertEqual(args.dataset, "voc20")
        self.assertIsNone(args.subset_manifest)

    def test_cli_accepts_distributed_flags(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--dataset",
                "voc",
                "--output-dir",
                "/tmp/probe-out",
                "--launcher",
                "pytorch",
                "--dist-backend",
                "nccl",
                "--gpu-collect",
                "--local-rank",
                "3",
            ]
        )
        self.assertEqual(args.launcher, "pytorch")
        self.assertEqual(args.dist_backend, "nccl")
        self.assertTrue(args.gpu_collect)
        self.assertEqual(args.local_rank, 3)

    def test_build_voc20_cfg(self):
        cfg = build_supervised_probe_cfg(
            model_key="deeplabv3plus_r50_d8",
            dataset_key="voc20",
            data_root="/tmp/VOCdevkit/VOC2012",
            work_dir="/tmp/probe-work",
            seed=0,
            max_iters=10,
            crop_size=512,
            samples_per_gpu=2,
            workers_per_gpu=1,
            val_workers_per_gpu=1,
        )
        self.assertEqual(cfg.dataset_type, "PascalVOCDataset20")
        self.assertEqual(cfg.model.decode_head.num_classes, 20)
        self.assertEqual(cfg.data.train["type"], "PascalVOCDataset20")
        self.assertEqual(cfg.data.train["split"], "ImageSets/Segmentation/train.txt")
        self.assertEqual(cfg.data.val["split"], "ImageSets/Segmentation/val.txt")
        self.assertEqual(os.path.abspath("/tmp/VOCdevkit/VOC2012"), cfg.data_root)

    def test_build_cityscapes_cfg(self):
        cfg = build_supervised_probe_cfg(
            model_key="deeplabv3plus_r50_d8",
            dataset_key="cityscapes",
            data_root="/tmp/cityscapes",
            work_dir="/tmp/probe-work",
            seed=0,
            max_iters=10,
            crop_size=512,
            samples_per_gpu=2,
            workers_per_gpu=1,
            val_workers_per_gpu=1,
        )
        self.assertEqual(cfg.dataset_type, "CityscapesDataset")
        self.assertEqual(cfg.model.decode_head.num_classes, 19)
        self.assertEqual(cfg.data.train["img_dir"], "leftImg8bit/train")
        self.assertEqual(cfg.data.val["ann_dir"], "gtFine/val")
        self.assertEqual(os.path.abspath("/tmp/cityscapes"), cfg.data_root)

    def test_build_cfg_preserves_syncbn_for_distributed(self):
        cfg = build_supervised_probe_cfg(
            model_key="deeplabv3plus_r50_d8",
            dataset_key="voc",
            data_root="/tmp/VOCdevkit/VOC2012",
            work_dir="/tmp/probe-work",
            seed=0,
            max_iters=10,
            crop_size=512,
            samples_per_gpu=2,
            workers_per_gpu=1,
            val_workers_per_gpu=1,
            preserve_syncbn=True,
        )
        self.assertEqual(cfg.model.backbone.norm_cfg["type"], "SyncBN")
        self.assertEqual(cfg.model.decode_head.norm_cfg["type"], "SyncBN")
        self.assertEqual(cfg.model.auxiliary_head.norm_cfg["type"], "SyncBN")

    def test_build_voc_cfg_prefers_train_aug_when_available(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "ImageSets" / "Segmentation").mkdir(parents=True, exist_ok=True)
            (root / "SegmentationClassAug").mkdir(parents=True, exist_ok=True)
            (root / "ImageSets" / "Segmentation" / "train_aug.txt").write_text("2008_000001\n")
            cfg = build_supervised_probe_cfg(
                model_key="deeplabv3plus_r50_d8",
                dataset_key="voc20",
                data_root=tmpdir,
                work_dir="/tmp/probe-work",
                seed=0,
                max_iters=10,
                crop_size=512,
                samples_per_gpu=2,
                workers_per_gpu=1,
                val_workers_per_gpu=1,
            )
            self.assertEqual(cfg.data.train["ann_dir"], "SegmentationClassAug")
            self.assertEqual(cfg.data.train["split"], "ImageSets/Segmentation/train_aug.txt")


if __name__ == "__main__":
    unittest.main()
