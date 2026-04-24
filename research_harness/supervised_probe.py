from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist
from mmseg.apis import multi_gpu_test, single_gpu_test, train_segmentor
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

from segmentation import datasets as _segmentation_datasets  # noqa: F401
from slice_remix.manifests import SubsetManifest, load_subset_manifest
from validation_acceleration import build_validation_payload, subset_dataset_by_basenames


MMSEG_CONFIG_ROOT = "/home/yuhe/.conda/envs/clipdino2/lib/python3.9/site-packages/mmseg/.mim/configs"
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


@dataclass(frozen=True)
class ProbeDatasetSpec:
    key: str
    dataset_type: str
    num_classes: int | None
    reduce_zero_label: bool
    default_roots: tuple[str, ...]
    train_cfg: dict[str, Any]
    val_cfg: dict[str, Any]
    config_relpath: str


def _existing_root(*candidates: str) -> str:
    resolved = [os.path.abspath(str(candidate)) for candidate in candidates]
    for candidate in resolved:
        if os.path.exists(candidate):
            return candidate
    return resolved[0]


DATASET_SPECS: dict[str, ProbeDatasetSpec] = {
    "coco_stuff": ProbeDatasetSpec(
        key="coco_stuff",
        dataset_type="COCOStuffDataset",
        num_classes=171,
        reduce_zero_label=False,
        default_roots=(
            _existing_root(
                os.path.join(str(REPO_ROOT), "data", "coco_stuff164k"),
            ),
        ),
        train_cfg=dict(img_dir="images/train2017", ann_dir="annotations/train2017"),
        val_cfg=dict(img_dir="images/val2017", ann_dir="annotations/val2017"),
        config_relpath=os.path.join(
            "deeplabv3plus",
            "deeplabv3plus_r50-d8_512x512_80k_ade20k.py",
        ),
    ),
    "voc20": ProbeDatasetSpec(
        key="voc20",
        dataset_type="PascalVOCDataset20",
        num_classes=20,
        reduce_zero_label=True,
        default_roots=(
            _existing_root(
                os.path.join(str(REPO_ROOT), "data", "VOCdevkit", "VOC2012"),
                os.path.join(
                    str(WORKSPACE_ROOT),
                    "deeplab",
                    "research",
                    "deeplab",
                    "datasets",
                    "pascal_voc_seg",
                    "VOCdevkit",
                    "VOC2012",
                ),
            ),
        ),
        train_cfg=dict(
            img_dir="JPEGImages",
            ann_dir="SegmentationClass",
            split="ImageSets/Segmentation/train.txt",
        ),
        val_cfg=dict(
            img_dir="JPEGImages",
            ann_dir="SegmentationClass",
            split="ImageSets/Segmentation/val.txt",
        ),
        config_relpath=os.path.join(
            "deeplabv3plus",
            "deeplabv3plus_r50-d8_512x512_20k_voc12aug.py",
        ),
    ),
    "voc": ProbeDatasetSpec(
        key="voc",
        dataset_type="PascalVOCDataset",
        num_classes=21,
        reduce_zero_label=False,
        default_roots=(
            _existing_root(
                os.path.join(str(REPO_ROOT), "data", "VOCdevkit", "VOC2012"),
                os.path.join(
                    str(WORKSPACE_ROOT),
                    "deeplab",
                    "research",
                    "deeplab",
                    "datasets",
                    "pascal_voc_seg",
                    "VOCdevkit",
                    "VOC2012",
                ),
            ),
        ),
        train_cfg=dict(
            img_dir="JPEGImages",
            ann_dir="SegmentationClass",
            split="ImageSets/Segmentation/train.txt",
        ),
        val_cfg=dict(
            img_dir="JPEGImages",
            ann_dir="SegmentationClass",
            split="ImageSets/Segmentation/val.txt",
        ),
        config_relpath=os.path.join(
            "deeplabv3plus",
            "deeplabv3plus_r50-d8_512x512_20k_voc12aug.py",
        ),
    ),
    "cityscapes": ProbeDatasetSpec(
        key="cityscapes",
        dataset_type="CityscapesDataset",
        num_classes=19,
        reduce_zero_label=False,
        default_roots=(
            _existing_root(
                os.path.join(str(REPO_ROOT), "data", "cityscapes"),
            ),
        ),
        train_cfg=dict(img_dir="leftImg8bit/train", ann_dir="gtFine/train"),
        val_cfg=dict(img_dir="leftImg8bit/val", ann_dir="gtFine/val"),
        config_relpath=os.path.join(
            "deeplabv3plus",
            "deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py",
        ),
    ),
}


@dataclass
class SupervisedProbeTiming:
    subset_size: int
    started_at: str
    finished_at: str
    train_seconds: float
    eval_seconds: float
    total_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "subset_size": int(self.subset_size),
            "started_at": str(self.started_at),
            "finished_at": str(self.finished_at),
            "train_seconds": round(float(self.train_seconds), 3),
            "eval_seconds": round(float(self.eval_seconds), 3),
            "total_seconds": round(float(self.total_seconds), 3),
        }


def _probe_dataset_spec(dataset_key: str, data_root: str | None = None) -> tuple[ProbeDatasetSpec, str]:
    try:
        spec = DATASET_SPECS[str(dataset_key)]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unsupported supervised probe dataset: {dataset_key} (supported: {supported})") from exc
    resolved_root = os.path.abspath(data_root) if data_root else os.path.abspath(spec.default_roots[0])
    return spec, resolved_root


def _mmseg_config_path(model_key: str, dataset_key: str) -> str:
    if str(model_key) != "deeplabv3plus_r50_d8":
        raise ValueError(f"Unsupported supervised probe model: {model_key}")
    spec, _ = _probe_dataset_spec(dataset_key)
    path = os.path.join(MMSEG_CONFIG_ROOT, spec.config_relpath)
    if not os.path.exists(path):
        raise FileNotFoundError(f"MMSeg config not found: {path}")
    return path


def _resolved_dataset_cfg(spec: ProbeDatasetSpec, data_root: str) -> tuple[dict[str, Any], dict[str, Any]]:
    train_cfg = copy.deepcopy(spec.train_cfg)
    val_cfg = copy.deepcopy(spec.val_cfg)

    if spec.key in {"voc20", "voc"}:
        train_aug_split = os.path.join(data_root, "ImageSets", "Segmentation", "train_aug.txt")
        train_aug_ann_dir = os.path.join(data_root, "SegmentationClassAug")
        if os.path.isfile(train_aug_split) and os.path.isdir(train_aug_ann_dir):
            train_cfg["ann_dir"] = "SegmentationClassAug"
            train_cfg["split"] = "ImageSets/Segmentation/train_aug.txt"

    return train_cfg, val_cfg


def _deepcopy_cfg_item(value: Any) -> Any:
    return copy.deepcopy(value)


def _replace_syncbn_with_bn(node: Any) -> Any:
    if isinstance(node, dict):
        updated = {}
        for key, value in node.items():
            updated[key] = _replace_syncbn_with_bn(value)
        if updated.get("type") == "SyncBN":
            updated["type"] = "BN"
        return updated
    if isinstance(node, list):
        return [_replace_syncbn_with_bn(item) for item in node]
    return node


def build_supervised_probe_cfg(
    *,
    model_key: str,
    dataset_key: str,
    data_root: str | None,
    work_dir: str,
    seed: int,
    max_iters: int,
    crop_size: int,
    samples_per_gpu: int,
    workers_per_gpu: int,
    val_workers_per_gpu: int,
    preserve_syncbn: bool = False,
) -> Config:
    spec, resolved_root = _probe_dataset_spec(dataset_key, data_root)
    train_cfg, val_cfg = _resolved_dataset_cfg(spec, resolved_root)
    cfg = Config.fromfile(_mmseg_config_path(model_key, dataset_key))
    cfg = copy.deepcopy(cfg)
    if not preserve_syncbn:
        cfg.model = _replace_syncbn_with_bn(cfg.model)

    crop_hw = (int(crop_size), int(crop_size))
    train_pipeline = _deepcopy_cfg_item(cfg.data.train.pipeline)
    test_pipeline = _deepcopy_cfg_item(cfg.data.val.pipeline)

    for step in train_pipeline:
        if step.get("type") == "LoadAnnotations":
            step["reduce_zero_label"] = bool(spec.reduce_zero_label)
        elif step.get("type") == "Resize":
            step["img_scale"] = crop_hw
            step["ratio_range"] = (0.5, 2.0)
        elif step.get("type") == "RandomCrop":
            step["crop_size"] = crop_hw
        elif step.get("type") == "Pad":
            step["size"] = crop_hw

    for step in test_pipeline:
        if step.get("type") != "MultiScaleFlipAug":
            continue
        step["img_scale"] = crop_hw
        for transform in step.get("transforms", []):
            if transform.get("type") == "Resize":
                transform["keep_ratio"] = True

    if spec.num_classes is not None:
        cfg.model.decode_head.num_classes = int(spec.num_classes)
    if "auxiliary_head" in cfg.model and cfg.model.auxiliary_head is not None:
        if spec.num_classes is not None:
            cfg.model.auxiliary_head.num_classes = int(spec.num_classes)

    cfg.dataset_type = str(spec.dataset_type)
    cfg.data_root = str(resolved_root)
    cfg.data.samples_per_gpu = int(samples_per_gpu)
    cfg.data.workers_per_gpu = int(workers_per_gpu)

    cfg.data.train = dict(
        type=str(spec.dataset_type),
        data_root=str(resolved_root),
        pipeline=train_pipeline,
        **train_cfg,
    )
    cfg.data.val = dict(
        type=str(spec.dataset_type),
        data_root=str(resolved_root),
        pipeline=test_pipeline,
        **val_cfg,
    )
    cfg.data.test = copy.deepcopy(cfg.data.val)

    cfg.runner.max_iters = int(max_iters)
    cfg.checkpoint_config.interval = int(max_iters)
    cfg.evaluation.interval = int(max_iters)
    cfg.evaluation.metric = "mIoU"
    cfg.evaluation.pre_eval = True
    cfg.log_config.interval = 20

    cfg.gpu_ids = [0]
    cfg.device = "cuda"
    cfg.seed = int(seed)
    cfg.work_dir = os.path.abspath(work_dir)

    meta_root = cfg.setdefault("probe_meta", {})
    meta_root["model_key"] = str(model_key)
    meta_root["dataset_key"] = str(dataset_key)
    meta_root["data_root"] = str(resolved_root)
    meta_root["crop_size"] = int(crop_size)
    meta_root["val_workers_per_gpu"] = int(val_workers_per_gpu)
    return cfg


def _manifest_basenames(manifest: SubsetManifest) -> list[str]:
    basenames = [os.path.basename(path) for path in manifest.sample_paths]
    unique = sorted(set(str(item) for item in basenames))
    if not unique:
        raise ValueError("Subset manifest resolved to zero basenames.")
    return unique


def _build_train_dataset(cfg: Config, manifest: SubsetManifest | None) -> object:
    train_dataset = build_dataset(cfg.data.train)
    if manifest is not None:
        keep_basenames = _manifest_basenames(manifest)
        subset_dataset_by_basenames(train_dataset, keep_basenames)
    if len(train_dataset) == 0:
        raise RuntimeError("Subset basenames filtered the training dataset down to zero samples.")
    return train_dataset


def _build_val_dataset(cfg: Config) -> object:
    return build_dataset(cfg.data.val)


def _evaluate_segmentor(model: torch.nn.Module, cfg: Config) -> dict[str, Any]:
    rank, world_size = get_dist_info()
    distributed = bool(world_size > 1)
    val_dataset = _build_val_dataset(cfg)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=int(cfg.probe_meta.get("val_workers_per_gpu", 2)),
        dist=distributed,
        shuffle=False,
        persistent_workers=bool(int(cfg.probe_meta.get("val_workers_per_gpu", 2)) > 0),
        pin_memory=False,
    )
    if distributed:
        wrapped = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        results = multi_gpu_test(
            wrapped,
            val_loader,
            pre_eval=True,
            gpu_collect=bool(cfg.probe_meta.get("gpu_collect", False)),
        )
        if rank != 0:
            return {}
    else:
        wrapped = MMDataParallel(model.cuda(), device_ids=[0])
        results = single_gpu_test(wrapped, val_loader, pre_eval=True)
    metrics = val_dataset.evaluate(results, metric="mIoU")
    return build_validation_payload(
        eval_results=metrics,
        classes=list(val_dataset.CLASSES),
        validation_mode="full",
        used_inference_mode=str(cfg.model.get("test_cfg", {}).get("mode", "whole")),
    )


def _set_random_seed(seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _is_main_process() -> bool:
    rank, _ = get_dist_info()
    return int(rank) == 0


def run_supervised_probe(
    *,
    model_key: str,
    dataset_key: str,
    data_root: str | None,
    subset_manifest_path: str | None,
    output_dir: str,
    result_name: str,
    seed: int,
    max_iters: int,
    crop_size: int,
    samples_per_gpu: int,
    workers_per_gpu: int,
    val_workers_per_gpu: int,
    launcher: str = "none",
    dist_backend: str = "nccl",
    gpu_collect: bool = False,
) -> dict[str, Any] | None:
    manifest = (
        load_subset_manifest(os.path.abspath(subset_manifest_path))
        if subset_manifest_path
        else None
    )
    os.makedirs(output_dir, exist_ok=True)
    distributed = str(launcher) != "none"
    if distributed:
        init_dist(str(launcher), backend=str(dist_backend))

    cfg = build_supervised_probe_cfg(
        model_key=model_key,
        dataset_key=dataset_key,
        data_root=data_root,
        work_dir=output_dir,
        seed=seed,
        max_iters=max_iters,
        crop_size=crop_size,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        val_workers_per_gpu=val_workers_per_gpu,
        preserve_syncbn=distributed,

    )
    _, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size) if distributed else [0]
    cfg.device = "cuda"
    cfg.probe_meta["launcher"] = str(launcher)
    cfg.probe_meta["dist_backend"] = str(dist_backend)
    cfg.probe_meta["gpu_collect"] = bool(gpu_collect)

    started_at = datetime.now(timezone.utc).isoformat()
    total_start = time.perf_counter()
    _set_random_seed(seed)

    train_dataset = _build_train_dataset(cfg, manifest)
    model = build_segmentor(cfg.model)
    model.init_weights()
    model.CLASSES = getattr(train_dataset, "CLASSES", None)

    train_start = time.perf_counter()
    train_segmentor(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        validate=False,
        timestamp=datetime.now().strftime("%Y%m%d-%H%M%S"),
        meta={
            "seed": int(seed),
            "dataset_key": str(dataset_key),
            "data_root": str(cfg.probe_meta.get("data_root", "")),
            "subset_manifest": os.path.abspath(subset_manifest_path) if subset_manifest_path else None,
            "subset_size": int(len(train_dataset)),
            "model_key": str(model_key),
        },
    )
    train_seconds = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    eval_payload = _evaluate_segmentor(model, cfg)
    eval_seconds = time.perf_counter() - eval_start
    total_seconds = time.perf_counter() - total_start
    finished_at = datetime.now(timezone.utc).isoformat()
    if not _is_main_process():
        return None
    timing = SupervisedProbeTiming(
        subset_size=len(train_dataset),
        started_at=started_at,
        finished_at=finished_at,
        train_seconds=train_seconds,
        eval_seconds=eval_seconds,
        total_seconds=total_seconds,
    )

    result_payload = {
        "model_key": str(model_key),
        "dataset_key": str(dataset_key),
        "data_root": str(cfg.probe_meta.get("data_root", "")),
        "candidate_id": str(manifest.candidate_id) if manifest and manifest.candidate_id is not None else None,
        "subset_manifest": os.path.abspath(subset_manifest_path) if subset_manifest_path else None,
        "full_train_split_used": bool(manifest is None),
        "seed": int(seed),
        "config": {
            "max_iters": int(max_iters),
            "crop_size": int(crop_size),
            "samples_per_gpu": int(samples_per_gpu),
            "workers_per_gpu": int(workers_per_gpu),
            "val_workers_per_gpu": int(val_workers_per_gpu),
            "launcher": str(launcher),
            "dist_backend": str(dist_backend),
            "gpu_collect": bool(gpu_collect),
        },
        "subset_size": int(len(train_dataset)),
        "metrics": eval_payload,
        "timing": timing.to_dict(),
    }
    if str(dataset_key) == "coco_stuff":
        result_payload["coco_stuff"] = eval_payload

    result_path = os.path.join(output_dir, result_name)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)
    return result_payload
