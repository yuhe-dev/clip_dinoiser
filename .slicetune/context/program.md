# 研究计划总览

本文件融合了：

- 项目背景
- 当前阶段
- 当前研究计划

目标是让新会话先用一个文件快速建立“项目是什么、现在在哪、接下来干什么”的整体理解。

---

## 1. 项目身份

本仓库当前被作为 **SliceTune** 的后端研究锚点来推进。

它不是：

- 通用前端产品
- 单纯 benchmark 仓库
- 只做离线推荐的系统

它当前主要承载的是一个研究问题：

> 如何基于可解释的 feature / slice 表达训练数据分布，并在固定预算下进行可控的数据重配、真实训练验证与人机协同优化？

当前更明确的研究对象补充（经 `2026-04-19` 用户澄清后更新）：

- 当前项目**不以“寻找单调 feature 轴”**为主要目标
- 当前更核心的问题是：
  - 多种数据特征会以**耦合、非单调、连续分布**的方式共同影响训练结果
  - 过高或过低都可能不好
  - 更优状态往往来自若干 feature 的**适度混合分布**
- 因此当前系统目标不是：
  - 保证全局最优
  - 给出单轴排序式 recommendation
- 而是：
  - 在具体模型训练任务场景中
  - 帮助用户理解当前数据特征分布的适合侧重方向
  - 通过人机交互把训练分布逐步推向“更好但不必最优”的状态

---

## 2. 当前仓库与主锚点

当前主锚点：

- 仓库：`clip_dinoiser`
- 主 case：`image segmentation`
- 当前主 benchmark：`COCO-Stuff` 相关训练与验证流程

当前工作区布局（自 `2026-04-17` 起）：

- 工作区根目录：`/home/yuhe/slicetune`
- 主研究仓库：`/home/yuhe/slicetune/clip_dinoiser`
- 官方 DeepLab 代码：`/home/yuhe/slicetune/deeplab`
- 根目录共享 feature 包：`/home/yuhe/slicetune/feature`
- 为兼容旧绝对路径与历史 artifact，保留软链接：
  - `/home/yuhe/clip_dinoiser -> /home/yuhe/slicetune/clip_dinoiser`

虽然长期可扩展到更多 case，但当前仓库事实上的 strongest anchor 仍是 segmentation。

当前新的工程方向补充：

- `feature extraction` 相关代码不再默认长期绑定在 `clip_dinoiser` 仓库内部
- 当前正逐步将其收敛为 workspace-level shared asset
- 迁移策略采用：
  - 先 root-level package
  - 再 clip_dinoiser 兼容桥接
  - 最后逐步迁出更重的实现模块

当前新的实验执行方向补充（经 `2026-04-18` 调试后更新）：

- 官方 `~/slicetune/deeplab` 仍保留，但当前重新降级为：
  - reference repo
  - recipe / eval sanity lane
- 原因：
  - 已定位到 `TF1.15 + CUDA10.0/cuDNN7 + RTX4090` 的 GPU 训练会触发 cuDNN convolution buffer 越界写，导致后续 logits/loss NaN
  - 因此官方 TensorFlow DeepLab 当前不适合作为高频主训练底座
- 当前主执行底座重新切回：
  - `clip_dinoiser` 中基于 **MMSeg / modern PyTorch** 的 supervised probe runner
- 当前更合理的集成方向是：
  - 以 modern PyTorch DeepLabV3+ / mmseg config 为主训练评测入口
  - 把 SliceTune 的 feature / subset / intervention 逻辑接到 dataset / manifest / subset filtering 这一层
  - 先跑 full-split baseline，再进入 feature-aware data selection

---

## 3. 当前研究状态

项目已不处于“从零搭 pipeline”的阶段，而是进入 **研究收敛期**。

现状特征：

- feature pipeline 已有
- slice construction 已有
- remix / materialization / surrogate 原型已出现
- 但全局结果动态范围较窄，证据密度低于模块复杂度

因此当前不应默认继续往后段无限扩张，而应回到上游逐层审计。

---

## 4. 当前阶段

### 阶段名称

**Phase 1：Feature Signal Audit / Learner Sensitivity Audit**

### 当前唯一主问题

1. 当前 `quality / difficulty / coverage` 特征是否真的对应可重复的训练结果变化？
2. 当前 learner 是否对数据组成变化足够敏感？
3. 当前观察到的 feature response 更像：
   - 单调响应
   - 非单调最优区间
   - 还是多特征耦合下的混合分布效应？

### 当前必须优先做的事

- noise floor 审计
- learner sensitivity 审计
- feature intervention / ablation / permutation 审计
- 除 global `mIoU` 外的局部指标分层

### 当前不该优先做的事

- surrogate family 扩张
- 更复杂 beam / prior-graph 搜索
- 前端叙事 polish
- 多 case 统一叙事

### 当前阶段 gate

只有当下面问题得到至少阶段性确认，才允许正式进入下一阶段：

- 数据组成效应大于噪声地板
- 至少一类 feature 或 feature 组合有稳定响应
- 至少一种 learner 对数据组成表现出足够敏感性

---

## 5. 当前分支状态

### keep

- feature extraction 审计
- learner sensitivity 审计
- benchmark 契约锁定
- noise floor 量化

### limited keep

- slice construction baseline 对照
- materialization fidelity 审计

### park

- candidate generation algorithm 扩张
- prior graph / beam search 复杂化
- surrogate family 扩张
- frontend / explanation polish
- 多 case 统一叙事

---

## 6. 当前研究计划

### Cycle 1

- 建立 memory system 与 harness 骨架
- 建立工程规范与自动科研契约
- 汇总已有 noise floor 与结果范围证据

### Cycle 2

- 设计 learner sensitivity ladder
- 设计 feature intervention matrix
- 锁定 Judge 的 promote / reject 规则

### Cycle 3

- 判断当前 learner 是否仍适合作为主锚点
- 如有必要，触发 Literature Radar 和替代方法复现

---

## 7. 当前最重要的现实判断

当前最大风险不是“后段没人做”，而是：

- 响应信号太弱
- 导致后段复杂系统可能只是在拟合窄幅波动

因此，当前研究对象应优先收敛成：

- response-aware sliceability 审计框架
- 或 targeted slice/class repair 框架
- 或 human-in-the-loop 的 feature distribution steering / preference insight 框架

而不是先承诺一个强 recommendation / surrogate 系统。
