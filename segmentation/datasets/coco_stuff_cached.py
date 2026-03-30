from __future__ import annotations

import os

import numpy as np
from mmseg.datasets import DATASETS, CustomDataset

from slice_remix.eval_cache import load_cache_manifest
from .coco_stuff import COCOStuffDataset


@DATASETS.register_module(force=True)
class CachedCOCOStuffDataset(CustomDataset):
    CLASSES = COCOStuffDataset.CLASSES
    PALETTE = COCOStuffDataset.PALETTE

    def __init__(self, manifest_path: str, **kwargs):
        self.manifest_path = str(manifest_path)
        super().__init__(img_suffix=".jpg", seg_map_suffix=".npy", **kwargs)

    def _resolve_cache_path(self, relative_path: str) -> str:
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self.data_root, relative_path)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split=None):
        manifest_path = self._resolve_cache_path(self.manifest_path)
        rows = load_cache_manifest(manifest_path)
        img_infos = []
        for row in rows:
            basename = str(row["basename"])
            img_infos.append(
                {
                    "filename": basename,
                    "ori_filename": basename,
                    "cache_filename": str(row["image_npy"]),
                    "ori_shape": tuple(int(v) for v in row["ori_shape"]),
                    "img_shape": tuple(int(v) for v in row["cached_img_shape"]),
                    "scale_factor": tuple(float(v) for v in row.get("scale_factor", (1.0, 1.0, 1.0, 1.0))),
                    "ann": {"seg_map": str(row["mask_npy"])},
                }
            )
        return img_infos

    def get_gt_seg_map_by_idx(self, index):
        ann_info = self.get_ann_info(index)
        seg_map = ann_info["seg_map"]
        seg_map_path = self._resolve_cache_path(seg_map)
        return np.load(seg_map_path)

    def get_gt_seg_maps(self, efficient_test=None):
        for index in range(len(self)):
            yield self.get_gt_seg_map_by_idx(index)
