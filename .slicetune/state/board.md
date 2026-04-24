# 执行面总表

更新时间：2026-04-15（EXP-P1-004 Tier A 已完成，judge=`park`）

本文件融合了：

- 当前任务
- 实验队列
- 主分支状态

目标是让当前执行面只看一个文件。

---

## 1. 当前任务

### 已完成

#### TASK-005

- 名称：汇总已有 noise floor 证据并形成统一基线摘要
- 所属阶段：Phase 1
- 当前状态：已完成
- 结果摘要：
  - `EXP-P1-001` 已完成
  - global `mIoU`：`mean=24.2939`、`std=0.0260`、`range=0.14`

#### TASK-012

- 名称：跑通 `EXP-P1-002` 并量化固定 subset 的 training noise
- 所属阶段：Phase 1
- 当前状态：已完成并通过人类验收
- 结果摘要：
  - `seed 0-4` 全部完成
  - `mean=24.2860`、`std=0.0089`、`range=0.0200`
  - judge 结果：`promote`
  - 当前 card 状态：`completed`

### 进行中

#### TASK-008

- 名称：建立 agent-centered 的最小 `research_harness` 执行层
- 所属阶段：基础设施 / harness
- 当前状态：进行中
- 当前结果：
  - 已支持 `noise_floor`、`same_subset_multi_seed`、`learner_sensitivity_ladder`、`feature_intervention_matrix`、`literature_radar` 五条 loop 协议
  - 已支持 `run_manifest / judge_policy / progress.json / context_packet.json / task_plan.json`
  - 已支持 watchdog-ready queue runner
  - `feature_intervention_matrix` 已接入 `Tier A` executable runtime
- 下一步：在 `EXP-P1-004` 跑通后继续补 `Tier B` controls 与更强 lineage

#### TASK-009

- 名称：建立 v2 debate protocol 与独立 judge policy / run manifest 机制
- 所属阶段：基础设施 / harness
- 当前状态：进行中
- 当前结果：
  - 已支持 legacy/new debate schema 兼容
  - 已支持 daemon 自动生成 debate bundle
  - 已支持 `run_research_debate.py` 按 controller policy 校验最小轮数
  - 已支持 `Skeptic / Benchmark Steward / Literature Critic / Harness Reviewer` 四角审查
- 下一步：把后续 phase 的设计卡都迁入同一 debate 规范

#### TASK-010

- 名称：建立代码级 queue/controller gate
- 所属阶段：基础设施 / harness
- 当前状态：进行中
- 当前结果：
  - 已支持 `phase gate / debate gate / human review stop`
  - 已支持 `lease / heartbeat / stale reclaim / retry guard`
  - 已支持 human approval release
  - 已支持 design-only / executable loop 分离
- 下一步：继续强化 scientific planner loop

#### TASK-013

- 名称：建立 scheduler / attempt / daemon 层
- 所属阶段：基础设施 / harness
- 当前状态：进行中
- 当前结果：
  - 已支持 dependency-aware scheduler
  - 已支持 attempt manifest 与 runtime index
  - 已支持 detached supervisor + `auto-propose + auto-debate`
- 下一步：继续补更强的 canonical artifact lineage

#### TASK-014

- 名称：建立 phase-locked proposer / planning 层
- 所属阶段：基础设施 / harness
- 当前状态：进行中
- 当前结果：
  - 已支持静态 follow-on proposal
  - 已支持 dynamic literature radar materialization
  - 已修复 literature radar 被纯 harness 故障误触发的问题
  - `EXP-P1-003` 已在 `EXP-P1-002` 放行为 `completed` 后自动 materialize
  - `EXP-P1-004` 已在 `EXP-P1-003` 完成后自动 materialize
  - 已补 phase-locked proposal 的默认 `output_dir / debate_bundle_path`
- 下一步：在 `EXP-P1-004` 完成后验证下一轮 Phase 1 follow-on proposal 是否仍保持收敛

#### TASK-015

- 名称：引入 task-level progress 产物（`task_plan.json + progress.md`）
- 所属阶段：基础设施 / harness
- 当前状态：进行中
- 当前结果：
  - 已支持 task-level acceptance gate
  - 已支持 `task_board.json`
  - 已支持 research-native task 状态机
  - 已支持 taskflow 自动刷新 task-generated plan
- 下一步：推广到后续 Phase 1 loop

#### TASK-016

- 名称：将 harness 推进到 research-conductor 级自治闭环
- 所属阶段：基础设施 / harness
- 当前状态：进行中
- 当前结果：
  - 已新增 `loop_catalog`
  - 已新增 `context_packet`
  - 已支持 auto debate / auto propose / auto release
  - 已支持可选 tick watchdog
  - 已支持 auto-agentic planning / analysis / judgment artifact
  - 已支持 context-aware `judgment_brief`
  - 已支持真实联网 Literature Radar query / ranking / method card 产出
  - 已验证 `feature_intervention_matrix` 可以消费现有 `design_pack / evaluation_rubric / design_spec`
- 下一步：观察 `EXP-P1-004` 的真实执行轨迹，并在 Tier A 完成后验证自动衔接

#### TASK-017

- 名称：将 harness 从执行层推进为 agentic autoresearch cognition layer
- 所属阶段：基础设施 / harness
- 当前状态：进行中
- 当前结果：
  - 已支持 `hypothesis_brief / design_pack / evaluation_rubric / analysis_brief / judgment_brief`
  - 已支持 queue/daemon 自动回填上述 artifact
  - 已支持 literature query planning、OpenAlex retrieval、method cards 与 ranking
  - 已支持根据 `design_mode=minimal_learner_adaptability_audit` 编译具体 `hypothesis/design/rubric`
  - `feature_intervention_matrix` 已开始直接消费这些 artifact 进入真实执行
- 下一步：补更细的 `analysis_brief` / `judgment_brief` for learner adaptability outcomes

### 待启动

#### TASK-006

- 名称：设计 learner sensitivity ladder
- 所属阶段：Phase 1
- 当前状态：已完成
- 当前结果：
  - `EXP-P1-003` 已完成并 `promote`
  - `fast_cached_1ep=24.29`
  - `fast_1ep=20.39`
  - `standard_3ep=20.75`
  - 当前最优 regime 仍为 `feature_experiment_fast_cached_slide`
  - 当前更准确的解释应为：`protocol sensitivity established`
  - 当前尚未回答“哪种 learner algorithm 对数据组成变化更敏感”

#### TASK-007

- 名称：设计 feature intervention matrix
- 所属阶段：Phase 1
- 当前状态：`Tier A` 已完成，当前 card=`completed`，judge=`park`
- 当前结果：
  - `EXP-P1-004` 已生成
  - 已自动写出 `task_plan.json`
  - 已自动写出 `hypothesis_brief / design_pack / evaluation_rubric`
  - 已冻结 3 个 learner variants：
    - `L0_head_only`
    - `L1_task_head_plus`
    - `L2_last_block_partial`
  - 已冻结 2 条 probe axes：
    - `quality_sharpness`
    - `difficulty_small_object`
  - 已冻结 3 类 controls：
    - `real_feature_guided`
    - `shuffled_feature_guided`
    - `matched_random_control`
  - 已冻结 4 个 reporting metrics：
    - `composition_response_amplitude`
    - `response_to_noise_ratio`
    - `directional_consistency`
    - `feature_validity_advantage`
  - 已实现 `Tier A` runtime：
    - selective unfreeze
    - learner-specific noise floor
    - 两个 probe axes
    - `real_feature_guided`
  - 已修复 debate gate 与 preflight false-negative
  - 已完成 axis scoring、class-presence cache 与四个 `real_feature_guided` manifests 物化
  - 已定位并修复两层真实 worker 阻塞：
    - Python 3.9 联合类型注解兼容问题
    - Hydra struct 模式下 `trainable_modules` 注入失败
  - `Tier A` 全部运行已完成：
    - `3` 个 learner variants
    - `2` 条 probe axes
    - `6` 个 `real_feature_guided` learner-axis cells
  - learner-specific noise floor 全部一致：
    - `L0=24.29`
    - `L1=24.29`
    - `L2=24.29`
    - `std≈0`
  - feature-pair 响应在三个 learner 上也一致：
    - `quality_sharpness`：high=`23.73`，low=`23.88`，amplitude=`0.15`
    - `difficulty_small_object`：high=`24.15`，low=`24.28`，amplitude=`0.13`
  - `mean_off_target_drift_ratio=0.0154`
  - 当前正式结论不是“learner adaptability 已建立”，而是：
    - 当前 `L0/L1/L2` 这组三档可训练范围扩展没有拉开 composition sensitivity
    - 当前结果更像一次有效的 `Tier A` screen，而不是可 promote 的正结果
- 下一步：
  - 若继续验证 feature validity，则推进 `Tier B`，补 `shuffled_feature_guided + matched_random_control`
  - 若优先回答 learner 问题，则先做梯度路径审计，再设计更强 learner 分支；当前 `L0/L1/L2` 轻量 ladder 可能没有真正进入有效 loss 路径

#### TASK-018

- 名称：将 learner sensitivity 重新定义为 learner adaptability audit
- 所属阶段：Phase 1
- 当前状态：`Tier A` 已完成，当前解释为“当前 ladder 未能区分 learner”
- 当前目标：
  - 不再只比较 bundled training regime
  - 在主诊断 learner 之外，筛选少量值得接入的 open-vocabulary 历史 baseline 作为辅助比较
  - 当前 shortlist：
    - 第一优先级：`CLIP-DIY`
    - 第二优先级：`MaskCLIP / MaskCLIP+`
    - 第三优先级：`TCL`
    - 第四优先级：`GroupViT`
  - 当前不建议首波接入：
    - `ReCo`
    - `NamedMask`
    - `OVDiff`
    - `OVSegmentor`
    - `SegCLIP`
    - `ZeroSeg`
    - `CLIPpy`
  - 原因：
    - 当前仓库对 `MaskCLIP / TCL / GroupViT / FOUND` 已有明确代码血缘
    - `CLIP-DIY` 与当前方法同作者系谱、且依赖 FOUND，迁移成本最低
    - 其余方法更适合后续 robustness / external baseline，不适合作为当前第一波执行对象

#### TASK-019

- 名称：校准 supervised probe learner 并判断是否需要降级 `COCO-Stuff / image segmentation`
- 所属阶段：Phase 1
- 当前状态：进行中
- 当前结果：
  - `DeepLabV3+ R50-D8` 已接入手动训练链路
  - 首轮 `1000 iter` anchor：`mIoU=0.03`
  - 第二轮 `8000 iter` anchor：`mIoU=0.08`
  - 当前已定位出一个足以破坏 supervision 的协议错误：
    - 借用的 `ADE20K` train pipeline 中 `LoadAnnotations(reduce_zero_label=True)`
    - 对 `COCO-Stuff164k` 应为 `False`
  - 该 bug 已在 `research_harness/supervised_probe.py` 修复
  - 修复后重新跑 `DeepLabV3+`：
    - `8000 iter`
    - `mIoU=1.11`
    - `mAcc=2.64`
    - `aAcc=22.79`
  - 当前进一步确认：
    - 训练子集实际覆盖 `170 / 171` 类，并非“缺类导致 150 类全零”
    - 但长尾极重：
  - 当前新增进展（`2026-04-17`）：
    - 官方 TensorFlow DeepLab 最小链路已在 `~/slicetune/deeplab` 跑通
    - `model_test.py` 已通过
    - `VOC2012` 官方数据已下载并转换为 `TFRecord`
    - `local_test` 等价最小评测已得到：
      - `eval/miou_1.0_overall = 0.821973264`
    - 这说明官方 DeepLab 代码与服务器环境可作为可信 external reference
  - 当前主判断：
    - 官方 `deeplab` repo 已完成“可信基线/环境验真”职责
    - 且根据当前用户明确决策，后续 supervised lane 将优先以官方 DeepLab 框架作为主训练底座
  - 当前更合理的推进方式是：
      - 保留官方 `train.py / eval.py` 为主训练评测入口
      - 将 SliceTune 的 feature-aware subset / intervention 逻辑接入 dataset / split / TFRecord 层
      - 先围绕 `VOC20` 或 `Cityscapes` 做 benchmark-specific feature selection 实验
      - `clip_dinoiser` 内自实现 probe runner 降级为备用实现，不再是第一优先级主 lane
  - 当前 `VOC` lane 的具体协议：
      - 训练池优先采用 `train_aug`
      - 评测固定采用 `val`
      - 第一阶段先做固定预算子集：
        - `anchor / high / low / matched-random`
      - 不直接用 `trainval` 做本地主线诊断
      - `132` 类只出现在 `<=50` 张图
      - `150` 类只出现在 `<=100` 张图
    - 修复后 full-val 中约 `21` 类 IoU 非零，其非零类均值约 `9.05`
  - 当前解释更新为：
    - 训练已经不再是“完全坏掉”
    - 但 `COCO-Stuff-171 + 1000 random images + full-val` 对当前 supervised probe 仍过于高熵
    - 模型只学会少数常见大类 / stuff 类，`mIoU` 被长尾类严重拉低
- 下一步：
- 当前优先考虑降级 benchmark，而不是切换任务
- 候选顺序：
  - `COCO-Object`
  - `VOC20`
  - `Pascal Context59`
  - 本地核对补充：
    - `VOC20` 代码入口已在仓库中现成存在：
      - `segmentation/datasets/pascal_voc20.py`
      - `segmentation/configs/_base_/datasets/pascal_voc12_20.py`
    - 但当前本地 `data/VOCdevkit/VOC2012` 目录尚未准备好，因此不是零迁移成本切换
- 保留 `image segmentation` 主任务与当前 feature/slice 资产
- Literature Radar 当前新增四条并行关注主线：
  - `training dynamics / data maps`
  - `label quality / segmentation label error detection`
  - `model-aware subset selection / data valuation / data curation`
  - `distribution-shift / long-tail / multi-domain benchmark`
- 当前高层迁移判断：
  - 可以考虑彻底切换 learner 与 benchmark
  - 但更推荐的成功区间是：
    - `moderate-capacity supervised learner`
    - `small/medium budget`
    - `low-entropy + structured-shift segmentation benchmark`
  - 不推荐直接切到超强 foundation segmentation model 作为 Phase 1 诊断主 learner
  - 对外部建议的当前修正：
    - 高层方向基本认可：先找 `diagnostic benchmark–learner pair`
    - 但当前立即执行顺序应为：
      1. dataset-aware supervised probe runner
      2. 运行 `tools/convert_coco_object.py` 后切到 `COCO-Object`
      3. `Cityscapes/ACDC`
      4. `VOC20` 作为干净 sanity 候选
  - 当前新增纠正：
    - `COCO-Object` 转换脚本本地已现成存在：`tools/convert_coco_object.py`
    - README 中旧的 `tools/convert_coco.py` 是过期名字
    - 该脚本直接基于现有 raw `COCO-Stuff` PNG masks 生成 `*_instanceTrainIds.png`
    - 不需要额外 COCO instance JSON
- 改为比较不同 `trainable scope / adaptation mechanism` 对数据组成响应幅度的影响
- 优先围绕 `head-only -> task-head unfreeze -> partial backbone unfreeze/PEFT` 形成最小 ladder
- 当前原因：
  - 当前代码只训练 `obj_proj + bkg_decoder`
  - `clip_backbone / FOUND / DINO` 全冻结
  - 因此当前 learner 可能不是“太强”，而是“太僵”，无法吸收数据组成差异
- 下一步冻结原则：
  - 先使用现有 feature space 中少量 `probe feature axes`
  - 每个 learner 单独估计 noise floor
  - 引入 `real / shuffled / random` 对照，但必须保持 matched materialization
  - 第一轮不做大搜索，只做最小 factorial design
- 第一轮先固定为：
    - `3` 个 learner variants
    - `2` 条 probe axes
    - `Tier A -> Tier B -> Tier C` 三层执行计划
- 当前执行状态：
  - `EXP-P1-004` 已完成并正式写出：
    - `result_bundle.json`
    - `judge_report.json`
    - `analysis_brief.json`
    - `judgment_brief.json`
  - mechanical Tier A screen：
    - `real_cells_above_noise_floor_count=6`
    - `screen_passed=true`
  - 但 frozen rubric 未通过：
    - `real_beats_shuffled=false`
    - `real_beats_random=false`
    - 当前 judge=`park`
  - 更关键的科学观察：
    - `L0/L1/L2` 的 noise floor 与 cell response 都相同
    - 因此当前 selective-unfreeze 方案尚未表现出更强的数据组成敏感性
    - 并且需要警惕：当前默认 `feats_idx=-3` + hook `detach()` 可能让新增 trainable modules 没有真正参与当前 loss 的有效梯度传播
  - 当前已新增一条真正的 backbone-grad audit 分支：
    - 新配置：`feature_experiment_fast_cached_slide_backbone_grad`
    - 显式启用 `final` feature + `track_grad`
    - 已通过 smoke test 确认 `decode_head.proj / last block / ln_post` 均收到非零梯度

---

## 2. 实验队列

### completed

#### EXP-P1-001

- 主题：global noise floor summary
- 当前状态：completed

#### EXP-P1-002

- 主题：same subset multi training seeds
- 当前状态：completed
- runtime profile：`clipdino2`
- 当前产物：
  - `result_bundle.json`
  - `judge_report.json`
  - `run_manifest.json`
  - `task_plan.json`
  - `hypothesis_brief / design_pack / evaluation_rubric / analysis_brief / judgment_brief`

### planned_follow_on

#### EXP-P1-003

- 主题：learner sensitivity ladder
- 当前状态：completed
- runtime profile：`clipdino2`
- 当前产物：
  - `task_plan.json`
  - `hypothesis_brief / design_pack / evaluation_rubric / analysis_brief`
  - `judge_report.json / result_bundle.json / run_manifest.json`
  - `EXP-P1-003_bundle.json` 及四角 review cards
  - 三个 regime 的独立 run 目录与 completion sentinel
- 当前解释限制：
  - 该实验一次改变了 `validation/eval config + model/config bundle + effective budget`
  - 因此它更适合被解释为 `protocol sensitivity audit`
  - 还不能单独证明“哪种 learner algorithm 对数据组成最敏感”

### planned_follow_on

#### EXP-P1-004

- 主题：feature intervention matrix
- 当前状态：completed
- runtime profile：`clipdino2`
- 当前产物：
  - `task_plan.json`
  - `hypothesis_brief / design_pack / evaluation_rubric`
  - agentic 上下文快照
  - `.slicetune/experiments/EXP-P1-004_design_spec.md`
- 当前新增产物：
  - `preflight_report.json`
  - `axis_scores_summary.json`
  - `materialization_index.json`
  - `noise_floor_summary.json`
  - `cell_results.json`
  - `result_bundle.json`
  - `judge_report.json`
  - `agentic/analysis_brief.json`
  - `agentic/judgment_brief.json`
- 当前正式结果：
  - judge=`park`
  - `Tier A` 已完成
  - `6/6` real learner-axis cells 超过各自 learner noise floor
  - `screen_passed=true`，但 `promote_ready=false`
  - 三个 learner variants 没有出现可区分的响应差异
- 当前解释：
  - 这不是“feature 完全无效”
  - 也不是“learner adaptability 已成功建立”
  - 更准确地说：当前这组三档 learner 扩展没有改变对 probe feature shifts 的响应形态

---

## 3. 主分支状态

### keep

- Phase 1 noise floor / training noise 审计
- learner sensitivity ladder 审计
- learner adaptability audit 设计
- runtime preflight / queue / daemon / attempt / taskflow
- human review stop + release
- agentic planning / analysis / judgment artifact
- executable literature radar loop

### park

- surrogate 扩张
- candidate generation 搜索复杂化
- frontend polish

### 当前停机原因

- `EXP-P1-004` 已完成并正式 `park`
- 当前系统不是执行面阻塞，而是研究决策点：
  - 是先推进 `Tier B` controls，验证 real feature 是否优于 shuffled/random
  - 还是先扩 learner 分支，寻找比当前 `L0/L1/L2` 更有区分度的适应机制

### 新 learner family 准备状态

- 已新增一条独立的 `intermediate-grad learner family`
  - 配置：[feature_experiment_fast_cached_slide_intermediate_grad.yaml](/home/yuhe/clip_dinoiser/configs/feature_experiment_fast_cached_slide_intermediate_grad.yaml)
  - 逻辑：
    - 保持 `feats_idx=-3`
    - `detach_intermediate_train_feats=false`
    - `enable_clip_grad_for_training=true`
- 当前 family 内部 ladder：
  - `L0`: `obj_proj + bkg_decoder`
  - `L1`: `L0 + resblocks.-3`
  - `L2`: `L1 + resblocks.-4`
- 已通过随机输入单步反传验证梯度路径：
  - `resblocks.-3: 8/12 params_with_grad`
  - `resblocks.-4: 12/12 params_with_grad`
- 当前建议动作：
  - 已完成 4 张卡并行 anchor：
    - `L0 anchor seed0 = 24.29`
    - `L1 anchor seed0 = 24.20`
    - `L2 anchor seed0 = 23.52`
    - `L2 anchor seed1 = 23.52`
  - 当前解释：
    - 这条 family 技术上成立，但目前没有显示出“更强 trainable scope 带来更好 baseline 或更明显噪声特征”
    - 当前更值得讨论的是：是否切入一个更标准、更 data-sensitive 的 supervised segmentation probe learner
    - 当前综合判断：
    - `frozen dense feature probe (P0/P1)` 适合作为便宜 sanity test
    - `DeepLabV3+ R50-D8` 更适合作为下一条主 probe learner
    - `SegFormer MiT-B0` 适合作为后续架构鲁棒性复核，而不是第一优先级

### supervised probe learner 准备状态

- 已新增主 probe learner 训练入口：
  - [research_harness/supervised_probe.py](/home/yuhe/clip_dinoiser/research_harness/supervised_probe.py)
  - [run_supervised_probe_experiment.py](/home/yuhe/clip_dinoiser/run_supervised_probe_experiment.py)
- 当前支持：
  - `DeepLabV3+ R50-D8`
  - manifest-driven COCO-Stuff subset training
  - full validation
  - `result.json` 写出
- 当前状态：
  - CLI 已通过 `--help`
  - dataset 子集过滤 smoke 已通过，anchor manifest 过滤后训练集长度为 `1000`
  - 已修复单卡训练失败：
    - 根因是官方 config 默认 `SyncBN`
    - 当前手动 probe 入口已自动改写为 `BN`
  - `max_iters=1` smoke 已确认能够：
    - 进入真实训练迭代
    - 保存 checkpoint
    - 进入 full validation
- 当前下一步：
  - 先由用户手工重新运行第一条 `DeepLabV3+ R50-D8` anchor probe 命令
  - 跑完后读取 `result.json` 决定是否：
    - 固化为主 Phase 1 probe learner
    - 或进一步收敛迭代预算 / crop size / batch size
  - 最新结果：
    - `deeplabv3plus_r50_d8_anchor_seed0` 已完成
    - `mIoU=0.03`
    - 当前解释：
      - 这不是有效科学负结果
      - 更像是 supervised probe budget 过短：`1000` iters on `1000` images with batch `2` only covers about `2` epochs
      - 当前下一步应优先重新校准：
        - 更长 `max_iters`
        - 必要时重标定学习率

### 2026-04-16 补充澄清

- `DeepLabV3+ on COCO-Stuff-171` 当前应明确视为：
  - 借用标准 supervised segmentation 架构的本地诊断 probe
  - 而不是对原始 DeepLabV3+ 论文 benchmark recipe 的 faithful reproduction
- 当前实现事实：
  - [research_harness/supervised_probe.py](/home/yuhe/clip_dinoiser/research_harness/supervised_probe.py:1) 直接借用了 MMSeg 的 `deeplabv3plus_r50-d8_512x512_80k_ade20k.py`
  - 并手工改写为：
    - `COCOStuffDataset`
    - `171` 类
    - `1000` 图 manifest 子集训练
    - full COCO-Stuff validation
- 最新有效结果：
  - [deeplabv3plus_r50_d8_anchor_seed0_8k_fixlabel/result.json](/home/yuhe/clip_dinoiser/artifacts/manual_runs/deeplabv3plus_r50_d8_anchor_seed0_8k_fixlabel/result.json:1)
  - `mIoU=1.11`
  - `mAcc=2.64`
  - `aAcc=22.79`
  - `21` 个类别 IoU 非零
  - 非零类别平均 IoU 约 `9.05`
- 当前解释：
  - 模型已开始学习常见类与大面积 stuff，但长尾类几乎全部压成 `0`
  - 因而更像 `benchmark–learner–metric` 契约过于严厉，而不是 `DeepLabV3+` 架构本身失败

### COCO-Object 当前定位修正

- 本地独立目录 [data/coco_object](/home/yuhe/clip_dinoiser/data/coco_object) 已存在且图片与标注已基本齐全
- 因而：
  - `COCO-Object` 不再应被视为“必须先做标注转换才能进入实验”的 benchmark
  - 当前更应聚焦于：
    1. 将 supervised probe 重构为 dataset-aware runner
    2. 直接支持 `COCOObjectDataset`
    3. 用 `COCO-Object` 作为下一条 supervised diagnostic benchmark

### benchmark 难度与比较口径澄清

- `VOC20`：
  - 当前更应视为最干净的 supervised sanity benchmark
  - 优势不是“最终最强 signal”，而是：
    - 类别数低
    - object-centric
    - 更容易先证明 learner 可以稳定吃到 signal
- `Cityscapes`：
  - 不只是“更简单”
  - 更重要的是：
    - 类别熵低于 `COCO-Stuff-171`
    - 场景结构高度规则
    - 更容易让天气/夜间/能见度/小目标这类分布效应显性化
- `CLIP-DINOiser` vs `DeepLabV3+`：
  - 不应按同一游戏规则直接比较论文数字
  - 当前更正确的 framing 是：
    - `DeepLabV3+`：fully supervised closed-set segmentation baseline
    - `CLIP-DINOiser`：open-vocabulary / weakly supervised / frozen-CLIP family
  - 因而：
    - `CLIP-DINOiser` 在 `VOC20`、`Cityscapes` 上显著低于 `DeepLabV3+` 的论文数字，并不意味着它“比 2018 老工作差所以没价值”
    - 更意味着它们回答的是不同难度与不同监督设定的问题

### 开放词汇 segmentation 的当前定位

- 开放词汇 / frozen-foundation segmentation 仍然是更强最终主张候选
- 但当前不应继续承担 Phase 1 第一性证明职责
- 原因不是它“不好”，而是：
  - learner 先验太强
  - 训练路径更间接
  - supervision 更弱
  - 一旦失败，几乎无法干净区分：
    - feature 是否无效
    - benchmark 是否压平信号
    - 还是 learner 本身对数据组成不敏感
- 当前更稳的路线：
  1. 先在 supervised diagnostic benchmark 上证明 `real > shuffled/random`
  2. 再把 protocol 回接到 `CLIP-DINOiser` / open-vocabulary family

### 2026-04-17 工作区布局更新

- 当前统一工作区根目录已调整为：
  - `/home/yuhe/slicetune`
- 当前并列仓库：
  - 主研究仓库：`/home/yuhe/slicetune/clip_dinoiser`
  - 官方 DeepLab 代码：`/home/yuhe/slicetune/deeplab`
- 兼容层：
  - 保留软链接 `/home/yuhe/clip_dinoiser -> /home/yuhe/slicetune/clip_dinoiser`
- `deeplab` 获取方式：
  - 通过官方仓库 `tensorflow/models` sparse clone
  - 当前工作树只展开 `research/deeplab`

### 官方 DeepLab 复现流程现状

- 已完成对官方 `DeepLab` 复现文档与主代码入口的首次审阅
- 当前确认：
  - 官方 repo 的“复现论文结果”不是一个单命令流程
  - 需要按 benchmark 拆成：
    1. 安装 TensorFlow 与依赖
    2. 配置 `PYTHONPATH`
    3. 准备数据并转换成 `TFRecord`
    4. 选择并下载初始 checkpoint 或作者发布 checkpoint
    5. 运行 `train.py`
    6. 运行 `eval.py`
    7. 视需要运行 `vis.py / export_model.py`
- 当前定位：
  - 后续若真的要复现官方数字，应优先从 `PASCAL VOC` 或 `Cityscapes` 按文档原配方进入
  - 不应把当前本地 MMSeg `DeepLabV3+ probe` 误认为官方 TensorFlow 实现的直接复现

### 官方 DeepLab 环境自检现状

- `deeplab` conda 环境中的 `model_test.py` 已经跑通：
  - `Ran 5 tests ... OK (skipped=1)`
- 当前已确认的最小可运行依赖修复：
  - `tensorflow-gpu==1.15.5`
  - `protobuf==3.20.3`
  - `tf_slim`
  - `PYTHONPATH` 包含：
    - `.../research`
    - `.../research/slim`
- 当前已确认的工作树要求：
  - 不能只 sparse checkout `research/deeplab`
  - 还必须展开 `research/slim`
- 当前解释：
  - 这说明官方 TensorFlow DeepLab 代码链路并未坏掉
  - 真正卡住的是：
    - TF1 与现代 `protobuf` 的兼容性
    - sparse checkout 不完整
    - `tf_slim` 这个额外 pip 依赖
- 当前下一步：
  1. 先按官方 `local_test.sh` / `pascal.md` 走最小 PASCAL 端到端链路
  2. 再决定要不要进入完整 benchmark reproduction 或迁移到本项目自己的 Phase 1 诊断协议

### 官方 DeepLab 数据下载现状

- `local_test.sh` 在当前服务器上的首个失败点不是训练，而是数据下载
- 当前失败位置：
  - `deeplab/datasets/download_and_convert_voc2012.sh`
  - 其中硬编码下载：
    - `https://data.deepai.org/PascalVOC2012.zip`
- 当前症状：
  - IPv4 连接超时
  - IPv6 不可达
- 当前解释：
  - 这是旧镜像/网络可达性问题，不是官方 `DeepLab` TensorFlow 代码链本身坏掉
- 当前建议路径：
  1. 手工准备 `VOC2012` 原始数据
  2. 继续使用官方脚本中的后半段：
     - `remove_gt_colormap.py`
     - `build_voc2012_data.py`
  3. 不再反复依赖 `data.deepai.org` 作为唯一入口

### 官方 DeepLab VOC 下载脚本修复现状

- 已直接修复官方脚本：
  - `/home/yuhe/slicetune/deeplab/research/deeplab/datasets/download_and_convert_voc2012.sh`
- 当前变更：
  - 下载源切换到 PASCAL VOC 官方地址：
    - `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar`
  - 解压方式改为：
    - `tar -xf`
  - 数据根路径修正为：
    - `pascal_voc_seg/VOCdevkit/VOC2012`
- 当前意义：
  - 后续继续跑 `download_and_convert_voc2012.sh` 时，数据入口已经与官方 VOC 发布格式对齐
  - 这比继续依赖旧 `DeepAI` 镜像更稳，也更贴近原始 benchmark 数据契约

### 官方 DeepLab VOC 数据准备状态

- 当前 `PASCAL VOC 2012` 数据准备已完成
- 已确认存在：
  - `VOCdevkit/VOC2012/JPEGImages`
  - `VOCdevkit/VOC2012/SegmentationClass`
  - `VOCdevkit/VOC2012/SegmentationClassRaw`
  - `VOCdevkit/VOC2012/ImageSets/Segmentation/{train,val,trainval}.txt`
  - `tfrecord/{train,val,trainval}-00000..00003-of-00004.tfrecord`
- 当前统计：
  - `JPEGImages`: `17125`
  - `SegmentationClass`: `2913`
  - `SegmentationClassRaw`: `2913`
- 当前下一步：
  1. 准备初始 checkpoint
  2. 跑最小 train / eval
  3. 再决定是否继续完整官方 reproduction

### 官方 DeepLab `local_test` 现状

- 当前 `local_test.sh` 的最小 `PASCAL VOC` 评测链路已成功通过
- 关键结果：
  - `eval/miou_1.0_overall = 0.821973264`
- 对照：
  - `local_test.sh` 注释中的预期值是：
    - `mIOU=82.20%`
- 当前判断：
  - 官方 TensorFlow DeepLab 在本机上的：
    - checkpoint
    - 数据读取
    - train.py
    - eval.py
    链路都已经站住
- 当前最合理的下一步：
  1. 冻结当前环境与命令，作为“官方可运行 anchor”
  2. 决定是：
     - 继续做官方 `pascal.md` / `cityscapes.md` 的更完整复现
     - 还是把这条官方 DeepLab 链接回 SliceTune 的诊断协议

### supervised segmentation 数据特征文献结论

- 当前从 segmentation / data-centric 文献中提炼出的高优先级数据特征主轴：
  1. 标注质量 / 边界噪声
  2. 类别长尾与像素不平衡
  3. 小目标比例 / 对象尺度
  4. 图像质量与 adverse conditions
  5. 域组成 / 分布偏移
- 当前对 Phase 1 的直接启发：
  - 若目标是放大数据组成效应，不要只围绕通用 `quality/difficulty/coverage` 抽象词
  - 应优先选择能被 benchmark 元数据或可计算指标稳定刻画的轴，例如：
    - label-suspicion
    - class-frequency / rare-class coverage
    - small-object ratio
    - illumination / fog / night / rain
    - city / weather / domain id

### learner 对 feature 的敏感性判断标准

- 当前不再用“global mIoU 有没有变化”作为唯一判断
- 当前固定使用四个核心量：
  1. `response_amplitude`
     - `Δ_real = metric(high) - metric(low)`
  2. `response_to_noise_ratio`
     - `RNR = |Δ_real| / sigma_noise`
  3. `control_gap`
     - `CG = |Δ_real| - max(|Δ_shuffled|, |Δ_random|)`
  4. `directional_consistency`
     - 多 seed 下 `Δ_real` 的方向是否一致
- 当前推荐 gate：
  - `RNR > 1`
  - `CG > 0`
  - `directional_consistency` 高
- 当前 metric 契约：
  - `global mIoU` 保留
  - 但同时必须报告与 feature 对应的局部 metric：
    - `small-object mIoU`
    - `rare-class mIoU`
    - `present-class mIoU`

### 新 benchmark 的 feature extraction 迁移原则

- 当前不建议在切换 benchmark 时“重写整套特征算法”
- 当前更合理的工程拆分是：
  1. **通用 extractor 复用**
     - 图像级质量特征
     - 通用视觉 embedding / coverage embedding
  2. **dataset adapter 重写**
     - `sample_index` / `image_rel` / `annotation_rel` 数据契约
     - mask 解析与类别 id 协议
     - class-presence / rare-class coverage
     - small-object ratio 等 label-derived features
  3. **feature validation 分层执行**
     - schema / shape / range sanity
     - 手工 spot-check
     - 独立重算交叉验证
     - 分布 sanity
     - materialized high/low subset face-validity
- 当前原则：
  - 先把新 benchmark 的特征链做成 “adapter + validator”
  - 不在第一步就推倒现有 `quality / difficulty / coverage` 主体实现

### `VOC20` 迁移骨架当前进展

- 当前已补出第一版 `VOC20` feature-adapter 骨架：
  - [dataset_specs.py](/home/yuhe/slicetune/clip_dinoiser/feature_utils/data_feature/dataset_specs.py:1)
  - [sample_voc20_subset.py](/home/yuhe/slicetune/clip_dinoiser/tools/sample_voc20_subset.py:1)
- 当前 `run_feature_pipeline.py` 已新增 `--dataset-spec`
  - 可将 dataset-specific metadata 合并进 feature meta
  - 见 [run_feature_pipeline.py](/home/yuhe/slicetune/clip_dinoiser/run_feature_pipeline.py:21)
- 当前 `class_presence` 已支持：
  - `annotation_rels`
  - `annotation_suffix`
  - `reduce_zero_label`
  - `label_id_map`
  - 见 [class_coverage.py](/home/yuhe/slicetune/clip_dinoiser/slice_remix/class_coverage.py:10)
- 当前已通过的针对性测试：
  - `test_dataset_feature_specs.py`
  - `test_voc20_subset_sampling.py`
  - `test_slice_remix_class_coverage.py`
  - `test_run_feature_pipeline_cli.py`

### 根目录公共 feature 包当前进展

- 当前已在 `~/slicetune` 根下建立过渡期公共包骨架：
  - `/home/yuhe/slicetune/feature`
- 当前第一批真正迁出的公共实现：
  - `feature.features.dataset_specs`
- 当前状态：
  - 根目录包骨架已创建
  - `dataset_specs` 已从 bridge 升级为 root-level source-of-truth
  - `clip_dinoiser.feature_utils.data_feature.dataset_specs` 已改为兼容桥接
  - root import 与 repo import 都已验证可用

### 官方 DeepLab `VOC train_aug` 资产当前进展

- 当前已将 canonical `train_aug` 资产落到官方 DeepLab 期望位置：
  - `train_aug.txt`：
    - `/home/yuhe/slicetune/deeplab/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/ImageSets/Segmentation/train_aug.txt`
  - `SegmentationClassAug/`：
    - `/home/yuhe/slicetune/deeplab/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/SegmentationClassAug`
- 当前已完成的清理与检查：
  - 已删除 zip 自带的 `__MACOSX` 与 `._*` 垃圾文件
  - `train_aug.txt` 行数：`10582`
  - `train_aug` ids 缺失 mask 数：`0`
  - 当前 `SegmentationClassAug` 中 `png` 文件总数：`12031`
- 当前转换状态：
  - `SegmentationClassAugRaw/` 已开始并已完成生成
  - 当前 `SegmentationClassAugRaw` 中 `png` 文件总数：`12031`
  - 官方 `build_voc2012_data.py` 已完成，已将 `SegmentationClassAugRaw + *.txt splits` 物化为 TFRecord
  - 当前四套 split shard 数均为 `4`：
    - `train`
    - `val`
    - `trainval`
    - `train_aug`
- 当前解释：
  - `train_aug` 所需的 split 清单与增强标注已经到位
  - 当前仍未进入可训练状态的剩余 gate 是：
    1. 基于完整 `train_aug` 做 `anchor/high/low/random` 的 2000 图 feature-aware split
    2. 实现并提取 `rare_class_coverage`
    3. 用官方 `train.py --train_split=<custom_split>` 跑第一轮 `anchor_2000`
  - 当前官方 `data_generator.py` 已补最小兼容：
    - 若 `split_name` 不在内置白名单中，但 `dataset_dir` 下存在 `<split>-*.tfrecord`
    - 则允许作为 custom split 继续训练
  - 这意味着后续：
    - `anchor_2000_seed0`
    - `feature_high_*`
    - `feature_low_*`
    - `matched_random_*`
    都可以不改训练 recipe，直接走官方 `--train_split=<custom_split>`

### 官方 DeepLab `TF1 GPU runtime` 当前阻塞

- 当前新事实：
  - `anchor_2000_seed0` 的 custom split 训练命令已经可以启动
  - 但当前 `deeplab` 环境中的 `tensorflow-gpu==1.15.5` 还没有真正注册 GPU
- 当前实机诊断：
  - `tf.test.is_gpu_available()` 返回 `False`
  - `device_lib.list_local_devices()` 仅列出：
    - `CPU`
    - `XLA_CPU`
    - `XLA_GPU`
  - TensorFlow 日志明确报缺：
    - `libcudart.so.10.0`
    - `libcublas.so.10.0`
    - `libcufft.so.10.0`
    - `libcurand.so.10.0`
    - `libcusolver.so.10.0`
    - `libcusparse.so.10.0`
    - `libcudnn.so.7`
  - 同时输出：
    - `Cannot dlopen some GPU libraries`
    - `Skipping registering GPU devices...`
- 当前解释：
  - 当前官方 DeepLab 训练慢，不是模型本身就该这么慢
  - 而是 `TF1.15` 预编译 wheel 期待 `CUDA 10.0 + cuDNN 7` 运行时库，当前环境只暴露了现代驱动与 `/usr/local/cuda/lib64`
  - 因此当前训练主体仍主要在 CPU 上运行
- 当前下一步：
  1. 先尝试为 `deeplab` 环境补齐：
     - `cudatoolkit=10.0.130`
     - `cudnn=7.6.5`
  2. 再次验证：
     - `tf.test.is_gpu_available()`
     - `device_lib.list_local_devices()`
  3. 若仍失败，则将“官方 TF1 DeepLab 不适合作为高频实验底座”升级为正式决策

### 官方 DeepLab `TF1 GPU runtime` 已修复

- 当前最新事实：
  - 用户已在 `deeplab` 环境中成功安装：
    - `cudatoolkit=10.0.130`
    - `cudnn=7.6.5`
  - 并通过：
    - `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"`
    - 让 conda 环境内的老 CUDA 运行时优先于系统路径
- 当前验证结果：
  - `tf.test.is_gpu_available()` 返回 `True`
  - `device_lib.list_local_devices()` 已出现：
    - `/device:GPU:0`
    - `/device:GPU:1`
    - `/device:GPU:2`
    - `/device:GPU:3`
  - TensorFlow 日志已成功打开：
    - `libcudart.so.10.0`
    - `libcublas.so.10.0`
    - `libcufft.so.10.0`
    - `libcurand.so.10.0`
    - `libcusolver.so.10.0`
    - `libcusparse.so.10.0`
    - `libcudnn.so.7`
- 当前解释：
  - 官方 DeepLab 当前已不再卡在 TF1 GPU runtime 缺库问题
  - 当前主实验可以正式进入：
    - `anchor_2000_seed0` 的单卡 GPU baseline 训练
- 当前操作约束：
  - 后续每次进入 `deeplab` 环境运行官方训练前，都必须先执行：
    - `conda activate deeplab`
    - `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"`

### 官方 DeepLab GPU 训练当前仍存在数值稳定性问题

- 当前新事实：
  - 在 GPU runtime 修复后，官方 `train.py` 已能：
    - 识别 GPU
    - 创建 `/device:GPU:0`
    - 成功加载 `cudnn/cublas/cudart`
  - 但无论是：
    - 自定义 `anchor_2000_seed0`
    - 还是官方内置 `trainval`
    的最小训练 smoke，均出现：
    - 长时间停在 `global_step/sec: 0`
    - 或首步即 `Loss is inf or nan`
- 当前已排除项：
  - 自定义 split 名未被识别
  - `anchor_2000_seed0` 标签值域异常
  - VOC 标签协议脏值
  - 缺失老 CUDA/cuDNN runtime 库
- 当前更合理解释：
  - 问题已经不再主要来自数据协议
  - 更像是：
    - 官方 `TF1.15 + CUDA10.0/cuDNN7 + RTX4090(8.9)` 这一训练栈在反向传播阶段数值不稳定
  - 当前至少已不能假设“GPU runtime 修复 = 官方训练可稳定作为主实验底座”
- 当前下一步：
  1. 先做一次 CPU 对照 smoke，确认同一官方 `trainval` 命令在 CPU 上是否仍 NaN
  2. 若 CPU 正常、GPU 异常，则将 root cause 更正式收敛为：
     - 老 TF1 栈与 Ada GPU 训练兼容性问题

### 官方 DeepLab 训练问题已收敛到 cuDNN 卷积算法损坏

- 当前新增事实：
  - 真正的 **CPU-only** 1-step debug 训练已成功跑通：
    - `CUDA_VISIBLE_DEVICES=''`
    - `trainval`
    - `batch_size=1`
    - `fine_tune_batch_norm=false`
    - `base_learning_rate=1e-5`
  - CPU-only debug 输出显示：
    - 输入图像统计正常
    - 标签值域正常
    - `logit_min/logit_max` 有限
    - `pixel_loss` 有限
    - `loss` 有限
    - 训练可正常结束
  - 对应的 GPU 1-step / 2-step debug 训练显示：
    - 第一步可以得到有限 `loss`
    - 之后 TensorFlow 报：
      - `Detected cudnn out-of-bounds write in convolution buffer!`
      - `This is likely a cudnn bug`
    - 紧接着下一步 `logit_min/logit_max` 变为：
      - `3.40282347e+38`
      - `-3.40282347e+38`
    - 然后 `total_loss/loss = nan`
- 当前已收敛解释：
  - 不是：
    - TFRecord 构造错误
    - 标签值域错误
    - custom split 接入错误
    - 单纯的 GPU 缺库问题
  - 而是：
    - `TF1.15 + CUDA10.0/cuDNN7 + RTX4090(Ada)` 训练时选到的 cuDNN 卷积算法会发生 buffer 越界写，进而污染 GPU 状态并导致后续 logits/loss NaN
- 当前执行含义：
  - 官方 DeepLab 的 **CPU 参考训练 lane** 仍可用于诊断
  - 官方 DeepLab 的 **GPU 高频训练 lane** 当前不适合作为 SliceTune 主实验底座
  - 后续若继续坚持官方 DeepLab，只应：
    - 作为 reference / recipe / eval sanity lane
    - 或尝试更激进的底层规避手段（不再作为默认主线）

### modern PyTorch supervised probe 已重新接回主线

- 当前新增工程结果：
  - `research_harness/supervised_probe.py` 已从 COCO-Stuff 专用实现扩展为 dataset-aware runner
  - 当前已支持：
    - `coco_stuff`
    - `voc20`
    - `cityscapes`
  - 当前已支持：
    - full-train baseline（无 manifest）
    - manifest-defined subset 训练
  - `run_supervised_probe_experiment.py` 已新增：
    - `--dataset`
    - `--data-root`
    - optional `--subset-manifest`
- 当前设计含义：
  - 现代 PyTorch / mmseg 现重新成为 supervised segmentation lane 的主训练底座
  - 官方 TensorFlow DeepLab 保留为对照与参考，不再承担高频主实验职责
- 当前新增对齐修正：
  - `voc20` 在数据根目录中若检测到：
    - `ImageSets/Segmentation/train_aug.txt`
    - `SegmentationClassAug/`
    则会自动优先使用增强训练池，而不再默认退回 `train.txt`
  - 同时新增标准 `voc`（21 类、含背景）dataset key，用于更直接逼近官方 / mmseg 的 VOC recipe
- 当前最直接下一步：
  1. 先跑 `voc20` full-split baseline
  2. 再跑 `cityscapes` full-split baseline
  3. baseline 正常后再接：
     - `anchor`
     - `high`
     - `low`
     - `matched_random`

### modern PyTorch `voc` full baseline 已跑通

- 当前新增结果：
  - `dataset=voc`
  - `subset_size=10582`（已正确使用 `train_aug`）
  - `max_iters=20000`
  - `samples_per_gpu=8`
  - `mIoU=73.51`
  - `mAcc=86.48`
  - `aAcc=93.64`
  - 总耗时：
    - `train_seconds=8373.886`
    - `eval_seconds=32.975`
    - `total_seconds=8408.018`
    - 约 `2.34h`
- 当前解释：
  - 这说明 modern PyTorch / mmseg DeepLabV3+ 主线已经从“能不能正常训练”阶段，进入“如何压缩实验 wall-clock 并设计 feature intervention budget”阶段
  - 当前 `73.51` 虽仍低于更强公开 recipe 的 `80%+` 口径，但已经足以证明：
    - learner 能正常学起来
    - `VOC train_aug` lane 已可作为后续 feature-sensitive 诊断底座
- 当前最直接下一步：
  1. 不再把 `20k` 作为每个 feature cell 的默认预算
  2. 先设计：
     - `5k`
     - `10k`
     两级 screening budget
  3. 在 `voc` lane 上先做第一轮：
      - `anchor`
      - `high`
      - `low`
      - `matched_random`

### `voc train_aug` full baseline 之后的主线实验顺序已收敛

- 当前新增执行判断：
  - 既然 `voc + train_aug + full baseline` 已证明：
    - learner 能正常学起来
    - modern PyTorch / mmseg lane 可稳定训练
  - 下一步不应继续反复在 full `train_aug` 上堆更多 baseline
  - 而应回到 Stage 1 主问题：
    - train pool 的 feature 是否真的对应可重复的训练结果差异
- 当前推荐顺序：
  1. 先在完整 `train_aug` 候选池上提取目标 feature
  2. 基于该全池 feature 物化受控训练子集：
     - `anchor`
     - `high`
     - `low`
     - `matched_random`
  3. 先用低成本 DDP screening budget 做第一轮响应审计
  4. 只有筛出有信号的 feature 轴，才升到更高预算确认
- 当前默认首批 feature 轴：
  - `small_object_ratio`
  - `rare_class_coverage`
- 当前默认首批执行预算：
  - pilot：`1000 iter`
  - confirmatory：`5000 iter`
  - 当前最新对照结论：
    - 单卡 `1000 iter`：
      - `mIoU=53.13`
      - `train_seconds=427.59`
      - `total_seconds=462.597`
    - `2` 卡 DDP + `SyncBN` `1000 iter`：
      - `mIoU=52.10`
      - `train_seconds=283.651`
      - `total_seconds=308.855`
    - 当前解释：
      - `2` 卡 DDP 在 `1000 iter` 下与单卡精度基本对齐（`ΔmIoU≈1.03`）
      - 同时带来约 `1.5x` 的 wall-clock 提升
    - 当前执行含义：
      - `2` 卡 DDP + `1000 iter` 已足够作为第一轮 feature screening 的默认底座

### modern PyTorch supervised probe 已支持多卡分布式训练入口

- 当前新增工程结果：
  - `research_harness/supervised_probe.py` 已新增 distributed path
  - 当前已支持：
    - `launcher=none` 的单卡路径
    - `launcher=pytorch/slurm/mpi` 的 mmcv distributed init
  - 当前训练阶段已支持：
    - `train_segmentor(..., distributed=True)`
  - 当前评测阶段已支持：
    - `multi_gpu_test(...)`
    - rank0 汇总 `mIoU`
    - 非 rank0 worker 静默退出，不重复写 `result.json`
  - `run_supervised_probe_experiment.py` 已新增：
    - `--launcher`
    - `--dist-backend`
    - `--gpu-collect`
    - `--local-rank/--local_rank`
- 当前验证：
  - `tests/test_supervised_probe_dataset_cfg.py` 已补 distributed CLI 参数测试
  - `python -m unittest tests.test_supervised_probe_dataset_cfg` 通过
- 当前执行含义：
  - 当前 modern PyTorch lane 已具备从单卡切到 `torchrun` 多卡的代码基础
  - 后续若要提升 feature intervention 吞吐，优先使用：
    - 多卡 DDP
    - 再结合 `5k/10k` screening budget
  - 当前已知现实 blocker：
    - 首轮 `4` 卡 DDP smoke 失败的直接原因不是 runner 代码，而是机器上已有他人进程长期占用全部 `4` 张 GPU
    - 观测到进程：
      - `pid=2772903`
      - `python train_reasoning.py`
      - 用户：`wangjy`
    - 其中第 `3` 号卡几乎被占满，导致该 rank 在第一层卷积处报：
      - `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`
  - 当前更合理的 DDP smoke 路径应为：
    - 临时只用未严重拥塞的 `2` 或 `3` 张卡
    - 或等待该外部进程释放 GPU 后再做 `4` 卡对齐实验
  - 当前新增精度对齐修正：
    - `build_supervised_probe_cfg(...)` 已改为：
      - 单卡时仍将 `SyncBN -> BN`
      - distributed 路径中保留原始 `SyncBN`
    - 当前含义：
      - 先前 `2` 卡 `200 iter` smoke 的低精度，不应继续简单解读为 DDP 训练逻辑错误
      - 其中一部分 recipe gap 来自 distributed 路径仍在使用普通 `BN`
  - 当前最新 smoke 结果：
    - `2` 卡 DDP + `SyncBN` + `200 iter` + 总 batch=`8`
    - `mIoU=9.94`
    - `train_seconds=63.687`
    - `eval_seconds=22.949`
    - `total_seconds=89.307`
  - 当前解释：
    - 分布式路径已稳定跑通并能正常产出 `result.json`
    - 但在仅 `200 iter` 的极短训练预算下，保留 `SyncBN` 并未显著改变早期精度形态
    - 当前 `200 iter` 结果更适合解释为：
      - DDP 工程可行性验证
      - 而不是最终的 recipe 质量判断

### VOC `train_aug` feature-prep 代码已接通

- 当前新增工程结果：
  - 已新增包层服务：
    - `slice_remix/voc_feature_subsets.py`
  - 已新增薄 CLI：
    - `tools/prepare_voc_train_aug_feature_experiment.py`
- 当前能力：
  - 读取完整 `train_aug` 池：
    - `ImageSets/Segmentation/train_aug.txt`
    - `JPEGImages`
    - `SegmentationClassAug`
  - 计算两条首批 feature 轴：
    - `small_object_ratio`
    - `rare_class_coverage`
  - 写出：
    - `feature_table.jsonl`
    - `summary.json`
    - `manifest_index.json`
    - `manifests/*.json`
  - 物化的 subset family：
    - `anchor`
    - `high`
    - `low`
    - `matched_random`
- 当前验证：
  - `python -m unittest tests.test_voc_feature_subsets tests.test_supervised_probe_dataset_cfg`
    - `OK`
  - `python tools/prepare_voc_train_aug_feature_experiment.py --help`
    - 正常
- 当前执行含义：
  - 现在已经具备 full `train_aug` feature-prep 的可执行代码
  - 下一步可以直接生成第一版 VOC feature screening artifacts，而不需要继续补基础脚本

### `small_object_ratio` 首轮 pilot 已完成

- 当前结果：
  - `anchor_2000`：
    - `mIoU=52.24`
    - `mAcc=67.54`
    - `aAcc=88.46`
  - `high_2000`：
    - `mIoU=35.71`
    - `mAcc=51.11`
    - `aAcc=84.84`
  - `low_2000`：
    - `mIoU=33.72`
    - `mAcc=54.30`
    - `aAcc=82.59`
  - `matched_random_2000`：
    - `mIoU=54.75`
    - `mAcc=75.02`
    - `aAcc=87.77`
- 当前解释：
  - 当前 learner 对 subset 组成变化存在强响应：
    - `high/low` 相对 `anchor` 的幅度远大于 noise floor
  - 但当前 axis interpretation 不是“more small objects helps or hurts”：
    - `high` 和 `low` 都明显低于 `anchor`
    - `matched_random` 反而略高于 `anchor`
  - 当前更合理的解释是：
    - 当前 `small_object_ratio` 这条轴在现有 materialization 下引入了明显混杂变化
    - 响应信号更像“extreme subset construction hurt training”而非单调 feature effect
- 当前补充诊断：
  - `low` subset 的平均每图前景类数仅 `1.163`，低于：
    - `anchor=1.470`
    - `matched_random=1.485`
  - `high` subset 也发生了明显类存在模式偏移：
    - `person` presence 从 `anchor=0.363` 提高到 `0.485`
- 当前执行含义：
  - 这轮结果足以说明：
    - 训练协议对数据组成变化敏感
  - 但还不足以 promote：
    - `small_object_ratio` 是一个干净的、单调可解释的优化杠杆
  - 下一步优先：
    - 继续跑 `rare_class_coverage` pilot
    - 同时考虑收紧 `matched_random` 与同轴互斥约束

### 研究目标已进一步澄清为“非单调、多特征耦合、偏好引导”的 insight 系统

- 当前新增判断（来自用户澄清）：
  - 项目主目标不是寻找单调 feature 轴
  - 当前更重要的是：
    - 识别多特征耦合下的非单调可取区间
    - 理解不同任务场景中“什么样的 feature distribution 更合适”
    - 通过人机交互而非全自动最优搜索，为用户提供 distribution steering insight
- 当前对现有 `small_object_ratio` pilot 的重新解释：
  - `high/low` 都差而 `matched_random` 更好
  - 不应被简单读成“这条轴无效”
  - 它反而更符合：
    - 单轴极端推进往往会破坏整体混合分布
    - 更好的状态可能位于多特征平衡的中间区域
- 当前执行含义：
  - 后续实验分析不应再只按“单轴单调好/坏”来判
  - 应开始显式关注：
    - 非单调 response
    - 多轴耦合
    - 用户偏好如何映射到可接受的 feature distribution 调整

### VOC pilot 执行链新增更严格 control 与结果汇总工具

- 当前新增工程结果：
  - `slice_remix/voc_feature_subsets.py` 已更新：
    - `matched_random` 现在会和同轴 `high/low` 以及 `anchor` 保持互斥
    - `summary.json` 会额外写出同轴 overlap 统计
  - 已新增结果汇总脚本：
    - `tools/summarize_supervised_probe_results.py`
- 当前验证：
  - `python -m unittest tests.test_voc_feature_subsets tests.test_supervised_probe_dataset_cfg`
    - `OK`
  - `python tools/summarize_supervised_probe_results.py --help`
    - 正常
- 当前执行含义：
  - 之后若要继续 `rare_class_coverage` pilot，或重新跑更干净的 `small_object_ratio` pilot
  - 应先重新生成一版 feature-prep artifacts，让严格互斥的 `matched_random` 生效

### `rare_class_coverage` strict pilot 已完成

- 当前结果：
  - `anchor_2000`：
    - `mIoU=53.62`
    - `mAcc=70.95`
    - `aAcc=88.37`
  - `high_2000`：
    - `mIoU=40.53`
    - `mAcc=61.14`
    - `aAcc=84.18`
  - `low_2000`：
    - `mIoU=14.58`
    - `mAcc=23.07`
    - `aAcc=81.33`
  - `matched_random_2000`：
    - `mIoU=41.34`
    - `mAcc=60.37`
    - `aAcc=85.46`
- 当前解释：
  - `anchor` 明显优于 `high / low / matched_random`
  - 当前结果更接近：
    - balanced mixture 优于单轴偏置或被迫互斥后的 control
  - 但仍不能直接 promote 为“rare_class_coverage 是单独可优化主轴”：
    - `high` 与 `matched_random` 都显著低于 `anchor`
    - `high` 同时伴随：
      - `mean_fg_classes` 从 `1.470 -> 2.184`
      - `small_object_ratio` 从 `0.263 -> 0.355`
    - `low` 则几乎塌缩成低覆盖、低复杂度子集：
      - `mean_fg_classes=1.034`
      - `small_object_ratio=0.088`
- 当前执行含义：
  - 这轮结果支持：
    - 当前任务更像存在“中间或混合分布更优”的结构
    - 极端推进某一特征会显著伤害训练
  - 下一步不应继续只做极端 `high/low`
  - 应优先进入：
    - 中等幅度 perturbation
    - 两轴联合/局部网格
    - 围绕 `anchor` 的 preference-aware distribution steering

### 已新增最近实验事实汇总文档

- 当前新增文档：
  - `.slicetune/logs/2026-04-19_recent_experiment_fact_sheet.md`
- 当前文档范围：
  - 最近 VOC supervised probe 实验
  - 训练算法与实现路径
  - 评测协议
  - 训练协议与结果数值
  - 当前和历史自定义特征指标及其语义

### 下一阶段实现计划已收敛为 staged feature screening

- 当前计划不再按“单个特征一路深挖到底”推进
- 当前默认分为三层：
  1. coarse screening
  2. strict validation
  3. local interaction / shape experiment
- 当前建议冻结的候选特征池规模：
  - `10–12` 个候选特征
  - 其中主系统目标冻结约 `6` 个
- 当前工程执行方向：
  - 先把 VOC feature 实验从“手写两条轴”推广为 generic feature registry + generic subset materialization
  - 再用当前已验证的：
    - `2` 卡 DDP
    - `1000 iter`
    - `2000` 图 subset
    作为 coarse screening 默认预算
- 当前建议优先候选族：
  - quality：
    - `boundary quality`
    - `laplacian sharpness`
    - `noise_pca`
  - difficulty：
    - `small_object_ratio / scale distribution`
    - `crop_survival_score`
    - `shape complexity / fragmentation`
    - `hardness proxy`
  - coverage：
    - `rare_class_coverage`
    - `foreground_class_count / pixel class entropy`
    - `object-level semantic coverage`

### VOC feature-prep 已完成第一轮通用化重构

- 当前包层边界已调整为：
  - `slice_remix/voc_feature_prep/contracts.py`
    - 数据契约与 dataclass
  - `slice_remix/voc_feature_prep/dataset.py`
    - VOC `train_aug` 数据契约与记录读取
  - `slice_remix/voc_feature_prep/scoring.py`
    - feature axis registry 与 feature table 计算
  - `slice_remix/voc_feature_prep/service.py`
    - subset materialization 与 artifact 写出
  - `slice_remix/voc_feature_prep/__init__.py`
    - 公共 API 出口
- 当前兼容策略：
  - `slice_remix/voc_feature_subsets.py` 已降级为 backward-compatible bridge
- 当前 CLI：
  - `tools/prepare_voc_train_aug_feature_experiment.py`
  - 现在支持：
    - `--feature-axis`
- 当前已接入的可选 VOC feature axes：
  - 默认 screening 轴：
    - `small_object_ratio`
    - `rare_class_coverage`
  - 新增 mask-native 轴：
    - `foreground_class_count`
    - `pixel_class_entropy`
    - `foreground_area_ratio`
    - `foreground_component_count`
    - `component_fragmentation`
- 当前验证：
  - `python -m unittest tests.test_voc_feature_subsets tests.test_supervised_probe_dataset_cfg`
    - `OK`
  - `python tools/prepare_voc_train_aug_feature_experiment.py --help`
    - 正常
