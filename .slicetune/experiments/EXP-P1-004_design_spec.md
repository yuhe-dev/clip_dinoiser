# EXP-P1-004 设计细则

## 实验身份

- `experiment_id`: `EXP-P1-004`
- `phase`: `Phase 1`
- `design_mode`: `minimal_learner_adaptability_audit`
- `loop_kind`: `feature_intervention_matrix`

## 实验目标

本实验不是直接寻找“最优特征分布”，而是先回答：

1. 当前 `head-only` learner 是否过于僵硬，难以吸收数据组成差异。
2. 在扩大 learner 可训练范围后，已有 feature axes 是否会表现出更强、更稳定的训练响应。
3. 真实 feature-guided intervention 是否优于 `shuffled` / `matched-random` 对照。

## 冻结不变的条件

- 任务：`image segmentation`
- benchmark：`COCO-Stuff`
- validation：`full`
- teacher policy：固定 `FOUND / DINO` teacher，不在本轮改变 teacher target
- subset budget：每个 materialized subset 固定 `1000` 张图
- anchor subset manifest：`artifacts/surrogate_random_v1/manifests/rand_subset_s0145_t00.json`
- 主指标：`mIoU`
- 本轮只改变：
  - learner 可训练范围
  - feature-guided subset composition along probe axes
- 本轮不改变：
  - 任务定义
  - 主 benchmark
  - validation protocol
  - teacher target

## Learner Variants

### L0: `head-only`

- 训练模块：
  - `obj_proj`
  - `bkg_decoder`
- 角色：当前主 baseline

### L1: `task-head-plus`

- 训练模块：
  - `obj_proj`
  - `bkg_decoder`
  - `clip_backbone.decode_head.proj`
- 角色：最低成本地扩大任务头可塑性

### L2: `last-block-partial`

- 训练模块：
  - `obj_proj`
  - `bkg_decoder`
  - `clip_backbone.decode_head.proj`
  - `clip_backbone.backbone.visual.transformer.resblocks[-1]`
  - `clip_backbone.backbone.visual.ln_post`
- 角色：在不动 teacher 的前提下，提供受控的表征可塑性

## Probe Feature Axes

当前 feature space 被视为 `working hypothesis`，不是 ground truth。

### Axis A: `quality_sharpness`

- block：`quality`
- 目标字段：
  - `features.laplacian.summary.q50`
  - `features.laplacian.summary.low_sharpness_mass`
  - `features.laplacian.summary.q90`
- score：
  - `z(laplacian.summary.q50) - z(laplacian.summary.low_sharpness_mass)`
- 高方向解释：`sharper`
- 低方向解释：`blurrier`

### Axis B: `difficulty_small_object`

- block：`difficulty`
- 目标字段：
  - `features.small_ratio.summary.mass_small_extreme`
  - `features.small_ratio.summary.mass_small_mid`
- score：
  - `z(small_ratio.summary.mass_small_extreme)`
- 高方向解释：`more extreme small-object concentration`
- 低方向解释：`less extreme small-object concentration`

### Optional Axis C: `coverage_density`

- block：`coverage`
- 目标字段：
  - `features.knn_local_density.summary.density_score`
  - `features.knn_local_density.summary.nearest_distance`
- score：
  - `z(knn_local_density.summary.density_score)`
- 使用条件：只有在 Tier B 已出现可推广 signal 后才解锁

## Control Families

每个 learner x axis cell 不只跑真实 feature，还要跑两类对照：

### `real_feature_guided`

- 使用真实 axis score 进行 high/low subset materialization

### `shuffled_feature_guided`

- 对同一 axis score 在 image id 上打乱
- 保留 score 分布，但打断 feature 与样本内容之间的真实对应关系

### `matched_random_control`

- 不使用 axis score 排序
- 只满足与真实 intervention 同等级别的：
  - subset budget
  - class histogram 匹配
  - non-target axis 匹配

## Materialization 规则

对于每个 `learner x axis x control x seed`，生成一对 subset：

- `high`
- `low`

每个 subset 规模固定为 `1000`。

必须记录：

- `intended_target_delta`
- `realized_target_delta`
- `off_target_feature_drift`
- `class_histogram_drift`
- `coverage_drift`

必须满足：

- class histogram matching
- non-target axis matching
- budget fixed
- full validation

## Noise Floor 规则

每个 learner variant 都必须先单独估计自己的 training noise。

做法：

- 固定 anchor subset
- 使用 training seeds：`[0, 1, 2]`
- 记录 learner-specific：
  - `noise_mean`
  - `noise_std`
  - `noise_range`

后续所有 `response_to_noise_ratio` 都必须除以该 learner 自己的 `noise_std`。

## 指标定义

### `signed_response`

- `mIoU(high) - mIoU(low)`

### `composition_response_amplitude`

- `abs(signed_response)`

### `response_to_noise_ratio`

- `composition_response_amplitude / learner_noise_std`

### `directional_consistency`

- 同一 `learner x axis x control` 下，各 seed 的 `signed_response` 与多数方向一致的比例

### `feature_validity_advantage`

- `real_feature_guided.response_to_noise_ratio - max(shuffled_feature_guided.response_to_noise_ratio, matched_random_control.response_to_noise_ratio)`

## Tiered 执行计划

### Tier A: `screen`

- 目的：确认是否存在大于 learner noise 的初步信号
- learner noise seeds：`[0, 1, 2]`
- feature pair seeds：`[0]`
- control families：
  - `real_feature_guided`
- 推进条件：
  - `composition_response_amplitude > learner_noise_std`
  - 已记录 `realized_target_delta`
  - `off_target_drift_ratio <= 1.0`

### Tier B: `confirm`

- 目的：确认真实 feature 是否稳定优于对照
- feature pair seeds：`[0, 1, 2]`
- control families：
  - `real_feature_guided`
  - `shuffled_feature_guided`
  - `matched_random_control`
- 推进条件：
  - `response_to_noise_ratio >= 2.0`
  - `directional_consistency >= 0.67`
  - `real_feature_guided` 同时优于 `shuffled` 与 `matched-random`
  - full validation
  - teacher frozen

### Tier C: `finalize`

- 目的：只对最有希望的 cell 做高置信确认
- feature pair seeds：`[0, 1, 2, 3, 4]`
- 最多保留：
  - `2` 个 promoted learner-axis cells
- 可选解锁：
  - `coverage_density`

## 结果解释矩阵

### 情况 A

- 所有 learner 都不超过自身 noise floor
- 解释：
  - 现有 feature probes 可能缺乏可利用信号
  - 或当前任务/预算下组成效应仍太弱

### 情况 B

- 更高可塑 learner 首次明显超过 noise floor
- 解释：
  - 当前问题更像 learner 太僵，而不是 feature 必然错误

### 情况 C

- 真实 feature beats controls，但只在部分 axis 上成立
- 解释：
  - 这些 axis 值得 `keep`
  - 其余 axis 可以 `park` 或 `kill`

### 情况 D

- learner 更可塑，但噪声也显著变大
- 解释：
  - 后续更应寻找“最可控 learner”，不是盲目更强的 learner

## 当前工程落点

本轮下一步不是一次性接通全部 Tier，而是：

1. 先实现 `Tier A` runtime handler
2. 先支持 `L0/L1/L2`
3. 先支持两个 probe axes：
   - `quality_sharpness`
   - `difficulty_small_object`
4. 在 runtime 中显式写出：
   - learner-specific noise floor
   - realized intervention fidelity
   - Tier A promote / park 信号
