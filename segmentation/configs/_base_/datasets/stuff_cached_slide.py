_base_ = ["../custom_import.py"]

dataset_type = "CachedCOCOStuffDataset"
data_root = "./data/coco_stuff164k_eval_cache_slide"

test_pipeline = [
    dict(type="LoadNpyImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 448),
        flip=False,
        transforms=[
            dict(type="RandomFlip"),
            dict(type="ImageToTensorV2", keys=["img"]),
            dict(type="Collect", keys=["img"], meta_keys=["ori_shape", "img_shape", "pad_shape", "flip", "img_info"]),
        ],
    ),
]

data = dict(
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=".",
        ann_dir=".",
        manifest_path="manifest.jsonl",
        pipeline=test_pipeline,
    )
)

test_cfg = dict(mode="slide", stride=(224, 224), crop_size=(448, 448))
