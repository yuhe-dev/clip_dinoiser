# 最近实验事实汇总（截至 2026-04-19）

## 范围

- 仓库：`clip_dinoiser`
- 当前主 case：`image segmentation`
- 本文档只记录可从代码、配置和 artifact 中直接核对的事实

## 任务场景

- 任务类型：语义分割
- 当前主数据场景：PASCAL VOC 2012
- 当前主训练池：`train_aug`
- 当前主验证集：`val`
- 当前监督训练入口：
  - [run_supervised_probe_experiment.py](/home/yuhe/slicetune/clip_dinoiser/run_supervised_probe_experiment.py:1)
  - [supervised_probe.py](/home/yuhe/slicetune/clip_dinoiser/research_harness/supervised_probe.py:1)

## 训练算法与实现

### 训练框架

- 框架：PyTorch + MMSegmentation
- 当前主模型 key：`deeplabv3plus_r50_d8`
- 当前 VOC 使用的 mmseg config：
  - `/home/yuhe/.conda/envs/clipdino2/lib/python3.9/site-packages/mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_20k_voc12aug.py`

### 模型结构

- segmentor：`EncoderDecoder`
- backbone：`ResNetV1c`
- decode head：`DepthwiseSeparableASPPHead`
- auxiliary head：`FCNHead`
- VOC 实验的类别数设置：
  - `voc`：`21`
  - `voc20`：`20`

### 归一化层

- 单卡训练：在 [supervised_probe.py](/home/yuhe/slicetune/clip_dinoiser/research_harness/supervised_probe.py:108) 中将 `SyncBN` 替换为 `BN`
- 分布式训练：在 [supervised_probe.py](/home/yuhe/slicetune/clip_dinoiser/research_harness/supervised_probe.py:357) 中通过 `preserve_syncbn=distributed` 保留 `SyncBN`

### 优化器与学习率策略

- optimizer：
  - `SGD`
  - `lr=0.01`
  - `momentum=0.9`
  - `weight_decay=0.0005`
- lr schedule：
  - `policy=poly`
  - `power=0.9`
  - `min_lr=0.0001`
  - `by_epoch=False`
- runner：
  - `IterBasedRunner`

## 数据集与数据协议

### 数据根目录

- 当前 VOC root：
  - `/home/yuhe/slicetune/deeplab/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012`

### split 选择逻辑

- `voc` / `voc20` 默认 train split：
  - `ImageSets/Segmentation/train.txt`
- 若存在以下文件，则训练改用 `train_aug`：
  - `ImageSets/Segmentation/train_aug.txt`
  - `SegmentationClassAug/`
- 当前 recent VOC 实验实际使用：
  - `ImageSets/Segmentation/train_aug.txt`
  - `SegmentationClassAug`
- val split：
  - `ImageSets/Segmentation/val.txt`

### 训练数据 pipeline 改写

在 [build_supervised_probe_cfg(...)](/home/yuhe/slicetune/clip_dinoiser/research_harness/supervised_probe.py:120) 中做了以下覆盖：

- `Resize.img_scale = (crop_size, crop_size)`
- `Resize.ratio_range = (0.5, 2.0)`
- `RandomCrop.crop_size = (crop_size, crop_size)`
- `Pad.size = (crop_size, crop_size)`
- 测试阶段 `MultiScaleFlipAug.img_scale = (crop_size, crop_size)`
- 测试阶段 `Resize.keep_ratio = True`

## 评测协议

- 训练调用：
  - [train_segmentor(...)](/home/yuhe/slicetune/clip_dinoiser/research_harness/supervised_probe.py:382)
- 训练期间：
  - `validate=False`
- 评测调用：
  - 单卡：`single_gpu_test`
  - 分布式：`multi_gpu_test`
- 评测数据：
  - `val` 全量
- 评测指标：
  - `mIoU`
  - `mAcc`
  - `aAcc`
- 输出内容：
  - summary
  - per-class `IoU`
  - per-class `Acc`
- 当前 result payload 中的 inference mode：
  - `whole`

## 已执行训练协议

### 全量 `train_aug` baseline

1. 单卡 full baseline
   - artifact：
     - [artifacts/supervised_probe_voc_full_seed0_20k/result.json](/home/yuhe/slicetune/clip_dinoiser/artifacts/supervised_probe_voc_full_seed0_20k/result.json:1)
   - 配置：
     - dataset=`voc`
     - subset size=`10582`
     - max_iters=`20000`
     - crop_size=`512`
     - samples_per_gpu=`8`
   - 结果：
     - `mIoU=73.51`
     - `mAcc=86.48`
     - `aAcc=93.64`
     - `train_seconds=8373.886`
     - `total_seconds=8408.018`

2. 单卡 1000 iter 对照
   - artifact：
     - [artifacts/supervised_probe_voc_1000iter_single_b8/result.json](/home/yuhe/slicetune/clip_dinoiser/artifacts/supervised_probe_voc_1000iter_single_b8/result.json:1)
   - 配置：
     - dataset=`voc`
     - subset size=`10582`
     - max_iters=`1000`
     - crop_size=`512`
     - samples_per_gpu=`8`
     - launcher=`none`
   - 结果：
     - `mIoU=53.13`
     - `mAcc=70.20`
     - `aAcc=88.32`
     - `train_seconds=427.590`
     - `total_seconds=462.597`

3. 2 卡 DDP 1000 iter 对照
   - artifact：
     - [artifacts/supervised_probe_voc_1000iter_ddp_2gpu_b8_syncbn/result.json](/home/yuhe/slicetune/clip_dinoiser/artifacts/supervised_probe_voc_1000iter_ddp_2gpu_b8_syncbn/result.json:1)
   - 配置：
     - dataset=`voc`
     - subset size=`10582`
     - max_iters=`1000`
     - crop_size=`512`
     - samples_per_gpu=`4`
     - launcher=`pytorch`
     - dist backend=`nccl`
   - 结果：
     - `mIoU=52.10`
     - `mAcc=67.58`
     - `aAcc=88.65`
     - `train_seconds=283.651`
     - `total_seconds=308.855`

## 子集特征实验准备协议

### 代码与入口

- 特征表与 manifest 生成模块：
  - [slice_remix/voc_feature_subsets.py](/home/yuhe/slicetune/clip_dinoiser/slice_remix/voc_feature_subsets.py:1)
- CLI：
  - [tools/prepare_voc_train_aug_feature_experiment.py](/home/yuhe/slicetune/clip_dinoiser/tools/prepare_voc_train_aug_feature_experiment.py:1)

### 生成设置

- 数据池：
  - VOC `train_aug`
- pool size：
  - `10582`
- subset size：
  - `2000`
- anchor seed：
  - `0`
- candidate budget：
  - `8582`
- small object 阈值参数：
  - `small_object_tau_ratio=0.02`

### 产物

- `feature_table.jsonl`
- `summary.json`
- `manifest_index.json`
- `manifests/*.json`

### 生成的子集 family

- `anchor`
- `<axis>.high`
- `<axis>.low`
- `<axis>.matched_random`

### strict 版 feature-prep artifact

- 路径：
  - [artifacts/voc_train_aug_feature_prep_strict_seed0/summary.json](/home/yuhe/slicetune/clip_dinoiser/artifacts/voc_train_aug_feature_prep_strict_seed0/summary.json:1)
  - [artifacts/voc_train_aug_feature_prep_strict_seed0/manifest_index.json](/home/yuhe/slicetune/clip_dinoiser/artifacts/voc_train_aug_feature_prep_strict_seed0/manifest_index.json:1)
- strict 版 `matched_random` overlap 统计：
  - `small_object_ratio`：
    - `anchor=0`
    - `high=0`
    - `low=0`
  - `rare_class_coverage`：
    - `anchor=0`
    - `high=0`
    - `low=0`

## 已执行的 2000 图子集实验

### `small_object_ratio` pilot

artifact root：
- [artifacts/voc_small_object_pilot_seed0](/home/yuhe/slicetune/clip_dinoiser/artifacts/voc_small_object_pilot_seed0)

统一训练协议：
- dataset=`voc`
- subset size=`2000`
- max_iters=`1000`
- crop_size=`512`
- samples_per_gpu=`4`
- launcher=`pytorch`
- dist backend=`nccl`

结果：

| subset | result path | mIoU | mAcc | aAcc | train_seconds | total_seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| anchor | [result.json](/home/yuhe/slicetune/clip_dinoiser/artifacts/voc_small_object_pilot_seed0/voc_train_aug_anchor_2000_seed0/result.json:1) | 52.24 | 67.54 | 88.46 | 292.442 | 317.404 |
| small_object_ratio.high | [result.json](/home/yuhe/slicetune/clip_dinoiser/artifacts/voc_small_object_pilot_seed0/voc_train_aug_small_object_ratio_high_2000_seed0/result.json:1) | 35.71 | 51.11 | 84.84 | 290.465 | 315.121 |
| small_object_ratio.low | [result.json](/home/yuhe/slicetune/clip_dinoiser/artifacts/voc_small_object_pilot_seed0/voc_train_aug_small_object_ratio_low_2000_seed0/result.json:1) | 33.72 | 54.30 | 82.59 | 292.137 | 317.446 |
| small_object_ratio.matched_random | [result.json](/home/yuhe/slicetune/clip_dinoiser/artifacts/voc_small_object_pilot_seed0/voc_train_aug_small_object_ratio_matched_random_2000_seed0/result.json:1) | 54.75 | 75.02 | 87.77 | 289.961 | 315.085 |

### `rare_class_coverage` strict pilot

artifact root：
- [artifacts/voc_rare_class_pilot_strict_seed0](/home/yuhe/slicetune/clip_dinoiser/artifacts/voc_rare_class_pilot_strict_seed0)

统一训练协议：
- dataset=`voc`
- subset size=`2000`
- max_iters=`1000`
- crop_size=`512`
- samples_per_gpu=`4`
- launcher=`pytorch`
- dist backend=`nccl`

结果：

| subset | result path | mIoU | mAcc | aAcc | train_seconds | total_seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| anchor | [result.json](/home/yuhe/slicetune/clip_dinoiser/artifacts/voc_rare_class_pilot_strict_seed0/voc_train_aug_anchor_2000_seed0/result.json:1) | 53.62 | 70.95 | 88.37 | 289.728 | 314.512 |
| rare_class_coverage.high | [result.json](/home/yuhe/slicetune/clip_dinoiser/artifacts/voc_rare_class_pilot_strict_seed0/voc_train_aug_rare_class_coverage_high_2000_seed0/result.json:1) | 40.53 | 61.14 | 84.18 | 290.172 | 315.004 |
| rare_class_coverage.low | [result.json](/home/yuhe/slicetune/clip_dinoiser/artifacts/voc_rare_class_pilot_strict_seed0/voc_train_aug_rare_class_coverage_low_2000_seed0/result.json:1) | 14.58 | 23.07 | 81.33 | 292.770 | 318.006 |
| rare_class_coverage.matched_random | [result.json](/home/yuhe/slicetune/clip_dinoiser/artifacts/voc_rare_class_pilot_strict_seed0/voc_train_aug_rare_class_coverage_matched_random_2000_seed0/result.json:1) | 41.34 | 60.37 | 85.46 | 290.832 | 315.425 |

## 当前使用和记录过的特征指标与语义

### 早期 probe axes（EXP-P1-004 设计稿）

来源：
- [.slicetune/experiments/EXP-P1-004_design_spec.md](/home/yuhe/slicetune/clip_dinoiser/.slicetune/experiments/EXP-P1-004_design_spec.md:67)

1. `quality_sharpness`
   - block：`quality`
   - 目标字段：
     - `features.laplacian.summary.q50`
     - `features.laplacian.summary.low_sharpness_mass`
     - `features.laplacian.summary.q90`
   - score：
     - `z(laplacian.summary.q50) - z(laplacian.summary.low_sharpness_mass)`
   - 高方向解释：
     - `sharper`
   - 低方向解释：
     - `blurrier`

2. `difficulty_small_object`
   - block：`difficulty`
   - 目标字段：
     - `features.small_ratio.summary.mass_small_extreme`
     - `features.small_ratio.summary.mass_small_mid`
   - score：
     - `z(small_ratio.summary.mass_small_extreme)`
   - 高方向解释：
     - `more extreme small-object concentration`
   - 低方向解释：
     - `less extreme small-object concentration`

3. `coverage_density`（optional）
   - block：`coverage`
   - 目标字段：
     - `features.knn_local_density.summary.density_score`
     - `features.knn_local_density.summary.nearest_distance`
   - score：
     - `z(knn_local_density.summary.density_score)`

### 质量类实现

来源：
- [feature_utils/data_feature/implementations/quality.py](/home/yuhe/slicetune/clip_dinoiser/feature_utils/data_feature/implementations/quality.py:1)

1. `laplacian`
   - 类名：`LaplacianSharpness`
   - 标量分数：
     - 灰度图的 Laplacian 方差
   - 向量分数：
     - patch-wise Laplacian 方差

2. `bga`
   - 类名：`BoundaryGradientAdherence`
   - 标量分数：
     - mask 边界覆盖区域内的平均梯度幅值
   - 向量分数：
     - 边界像素位置的梯度幅值序列

3. `noise_pca_weak_texture`
   - 类名：`WeakTexturePCANoise`
   - 输出：
     - 基于弱纹理 patch + PCA 的噪声标准差估计

### 覆盖类实现

来源：
- [feature_utils/data_feature/implementations/coverage.py](/home/yuhe/slicetune/clip_dinoiser/feature_utils/data_feature/implementations/coverage.py:1)

1. `knn_local_density_faiss`
   - 类名：`KNNLocalDensityCLIPFaiss`
   - 输入：
     - 预计算 CLIP visual embedding
   - mode：
     - `mean_dist`
     - `inv_mean_dist`
     - `radius_count`

2. `prototype_margin` / prototype family
   - 同文件中定义了 prototype margin 相关 coverage 指标实现

### 困难度类实现

来源：
- [feature_utils/data_feature/implementations/difficulty.py](/home/yuhe/slicetune/clip_dinoiser/feature_utils/data_feature/implementations/difficulty.py:1)

1. `small_object_ratio`
   - 类名：`SmallObjectRatioCOCOStuff`
   - 计算对象：
     - 语义 mask 中目标类连通区域
   - 当前 VOC feature-prep 使用设置：
     - `thing_id_start=1`
     - `num_things=20`
     - `default_ignore_index=255`
     - `use_things_only=True`
     - `tau_ratio=0.02`
   - 当前标量分数：
     - `small_area / total_area`
   - 其中：
     - `tau = tau_ratio * H * W`
     - `small_area` 为面积小于 `tau` 的连通区域数量
     - `total_area` 为纳入统计的连通区域总数

2. `semantic_ambiguity_clip`
   - 类名：`SemanticAmbiguityCLIP`
   - 语义：
     - 以 CLIP 衡量 mask 区域视觉特征与标签文本之间的对齐程度

### 当前 VOC `train_aug` 实验专用特征轴

来源：
- [slice_remix/voc_feature_subsets.py](/home/yuhe/slicetune/clip_dinoiser/slice_remix/voc_feature_subsets.py:1)

1. `small_object_ratio`
   - 逐图分数：
     - 由 `SmallObjectRatioCOCOStuff.get_score(...)` 计算
   - 当前 strict feature-prep 中的 summary：
     - anchor mean=`0.2635`
     - high mean=`0.7489`
     - low mean=`0.0000`
     - matched random mean=`0.1469`

2. `rare_class_coverage`
   - 逐图分数：
     - 先用 [load_class_presence_matrix(...)](/home/yuhe/slicetune/clip_dinoiser/slice_remix/class_coverage.py:37) 读取每张图的前景类 presence 向量
     - 再计算全池 `class_presence_rate`
     - 再计算 `raw_weights = 1 / class_presence_rate`
     - 再做正值均值归一化
     - 每张图的 `rare_class_coverage` 为该图 presence 向量与 rarity weight 的加权和
   - 当前 strict feature-prep 中的 summary：
     - anchor mean=`1.0308`
     - high mean=`1.9062`
     - low mean=`0.3341`
     - matched random mean=`0.9444`

3. `foreground_class_count`
   - 逐图值：
     - presence 向量求和
   - 当前写入：
     - `feature_table.jsonl`

4. `present_classes`
   - 逐图值：
     - 当前图中出现的前景类别名称列表
   - 当前写入：
     - `feature_table.jsonl`

## 结果汇总工具

- 汇总脚本：
  - [tools/summarize_supervised_probe_results.py](/home/yuhe/slicetune/clip_dinoiser/tools/summarize_supervised_probe_results.py:1)
- 用途：
  - 读取一组 `result.json`
  - 相对某个 reference 输出 summary 和 delta
