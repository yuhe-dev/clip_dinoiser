# Phase 1 日志：Feature Signal Audit

## 2026-04-12

### 本次动作

- 建立第一版中文 `AGENTS.md`
- 建立 `.slicetune/` 目录骨架
- 建立 context / state / templates / handoff 初版文件
- 补充工程规范、代码结构契约与运行模型说明
- 补充模块边界、脚本治理、自动科研循环与证据晋升契约
- 补充包级开发契约、测试契约、变更控制契约与 Literature Radar 契约
- 将规范文件进一步收敛为 `program.md + playbook.md + board.md + decision_log.md`
- 建立 `research_harness/` 第一版薄执行层
- 新增机器可读 experiment card：`EXP-P1-001_noise_floor.json`
- 跑通第一条真实 tick：`python run_research_tick.py --experiment-card .slicetune/experiments/EXP-P1-001_noise_floor.json`
- 将 judge threshold 外置为独立 `judge_policy`
- 为 tick 自动补充 `run_manifest.json`
- 补充 `MEMORY.md` 与 proposer-reviewer-arbiter debate 模板
- 新增 `run_research_queue.py`
- 新增 `controller_policy.json` 与 `human_review.json`
- 将 `phase gate / debate gate / human review stop` 下沉为代码约束

### 当前判断

- 当前系统的主要问题不是功能缺失，而是证据链尚未按阶段冻结
- 当前必须先审计 response signal 与 learner sensitivity
- 下游模块暂不应继续扩大主线投入
- 当前全局 `mIoU` noise floor 已可结构化表述：`count=192`、`mean=24.2939`、`std=0.0260`、`range=0.14`
- 该结果支持“先做同 subset 多 training seed 与 learner sensitivity ladder，再谈下游搜索扩张”的判断
- 若要支撑多小时自主运行，下一步最关键的是 queue / resume / heartbeat，而不是继续堆新 prompt
- 当前 controller 已能完成单卡执行与 gate 检查，但还不具备真正的多小时 resume / heartbeat 能力

### 下一步建议

1. 将 `EXP-P1-002` 写成机器可读 experiment card
2. 扩展 `research_tick` 以支持同 subset 多 training seed loop
3. 为多小时运行补 `heartbeat / resume / multi-card scheduler`
4. 为 debate 产物补机器可读 bundle 规范
5. 设计 learner sensitivity ladder

## 2026-04-13

### 本次动作

- 对 `EXP-P1-002` 的首次失败做根因追踪，确认不是实验卡逻辑错误，而是 base env 缺失 `torchvision / mmcv`
- 探测本机已有 conda env，确认 `clipdino2` 具备：
  - `torch 1.12.1+cu116`
  - `torchvision 0.13.1+cu116`
  - `mmcv 1.6.0`
  - `hydra 1.3.2`
  - `omegaconf 2.3.0`
- 以 debate 形式审查 runtime / preflight / resume 设计，并将结论固化为 `RUNTIME-V1`
- 新增 runtime profile registry：`.slicetune/runtime/runtime_profiles.json`
- 新增 `research_harness/runtime_profiles.py` 与 `research_harness/preflight.py`
- 将 worker runtime 选择、依赖预检、CUDA 检查、config/script 存在性检查下沉为代码 gate
- 修复 controller 在 `run_research_tick` 抛异常时卡片停留在 `running` 的状态污染问题
- 为 `same_subset_multi_seed` 新增 `completion.json` 成功 sentinel，避免假 resume
- 重新排队并启动 `EXP-P1-002`，当前已在 `clipdino2` runtime 下通过 preflight 并开始真实训练
- `seed 0` 已真实完成，结果 `mIoU=24.29`，`train_seconds=21.984`，`eval_seconds=575.152`
- `seed 1-3` 已继续完成，当前都得到 `mIoU=24.29`
- `seed 4` 已接续启动
- 新增 scheduler / attempt / daemon 层，使系统具备：
  - dependency-aware 选卡
  - attempt manifest
  - retry 上限
  - daemon idle / continue / stop 基础入口
- 新增 absorber / runtime index 层，使系统可自动聚合当前 cards / attempts / judge 状态
- 新增 proposer / proposal policy 层，使系统可以根据 `runtime_index` 生成 phase-locked follow-on 提案
- 在 `EXP-P1-002` 尚未完成前，真实 proposer 只写出了空的 `proposal_index.json`，说明它没有越权提前扩张下一步主线

### 当前判断

- runtime / preflight 不是附属细节，而是长时自治研究循环的第一层可靠性边界
- 当前系统已经从“能跑一次 tick”推进到“能先选可用 runtime，再发 worker”
- `EXP-P1-002` 现在终于在正确环境中进入真实训练，这使 learner sensitivity 审计第一次具备真实可执行性
- 当前系统已具备基础的 `scheduler + attempt + daemon + absorber` 闭环
- 当前系统也已具备基础 proposer 层
- 但更强的 attempt 级 canonical output lineage 和更自由的 planner/generator 仍未完成

### 下一步建议

1. 等待 `EXP-P1-002` 完成并进入 judge / human review stop
2. 汇总固定 subset 多 training seed 的 `mIoU mean / std / range`
3. 用该结果决定 `EXP-P1-003` learner sensitivity ladder 的具体优先级
4. 将 runtime profile / preflight 机制推广到更多 worker loop
5. 等 `EXP-P1-002` 完成后验证 proposer 是否自动 materialize `EXP-P1-003`
6. 继续补更强的 attempt lineage

## 2026-04-13（补记）

### 本次动作

- 增加 task-level progress 产物：`task_plan.json + progress.md`
- `noise_floor` 与 `same_subset_multi_seed` 的 progress 写入已接通
- 进度产物将作为后续 debate / review / handoff 的统一落点
- debate gate 升级为独立验证模块，并支持可选 artifact 校验
- 新增 `run_research_debate.py` 作为 debate bundle 的统一组装/校验入口
- controller 增加 task-level acceptance gate（基于 `task_plan.json` 的 `acceptance_status`）
- 新增 task board 汇总与 daemon 自动更新（`task_board.json`）
- 采用研究版 task 状态机，并在 `task_plan.json` 中记录 `research_state / next_state`

### 当前判断

- 系统已开始从“卡片级执行”向“任务级 conductor”迁移
- 这能显著降低长时上下文漂移风险，且便于后续 reviewer/arbiter 做 per-task 验收

### 下一步建议

1. 让 daemon/queue 在每次 tick 后强制更新 task-level progress
2. 将 progress 产物推广到后续 loop（如 `learner_sensitivity_ladder`）

## 2026-04-14

### 本次动作

- 将 taskflow 继续推进为更接近 conductor 的版本：
  - 新增 `loop_catalog`，区分 `design_only` 与 `executable` loop
  - 新增 `context_packet.json`
  - 新增 daemon 自动 debate 生成
  - 新增 human approval release
  - 新增 queue-level watchdog-ready tick execution
- 将 dynamic literature radar 收紧为只对研究级 retry exhaustion 触发
- 修复真实运行暴露出的 `run_research_tick.py` 中 `load_json` import 漏失
- 将 `EXP-P1-002` 从基础设施故障中恢复，重新入队并用更新后的 daemon 跑完
- `EXP-P1-002` 已自动写出：
  - `result_bundle.json`
  - `judge_report.json`
  - `run_manifest.json`
  - `task_plan.json`
- `EXP-P1-002` 已自动进入 `awaiting_human_review`

### 当前判断

- 当前 harness 已具备：
  - 代码级 task state machine
  - auto debate
  - auto propose
  - runtime preflight
  - heartbeat / reclaim / retry guard
  - context packet
  - human approval stop + release
- `EXP-P1-002` 的 training noise 结果支持继续推进 Phase 1：
  - `mean=24.2860`
  - `std=0.0089`
  - `range=0.0200`
  - `noise_to_global_floor_ratio=0.344`
- 这说明固定 subset 下的 training noise 小于 global random-subset floor，后续仍值得审计 learner sensitivity 与 feature intervention

### 下一步建议

1. 由人类审核 `EXP-P1-002`
2. 审核通过后让 daemon 自动释放该卡，并允许 proposer/materializer 继续后续分支
3. 将 `learner_sensitivity_ladder` 接成真正可执行 loop

## 2026-04-14（续）

### 本次动作

- 将 `agentic` 层正式接入真实 autonomous loop：
  - tick 执行前自动写 `hypothesis_brief / design_pack / evaluation_rubric`
  - tick 执行后自动写 `analysis_brief`
  - daemon 巡检时自动回填上述 artifact
- 新增 `agentic_judge`：
  - 基于 frozen rubric 的 `judge_contract`
  - 结合 `context_packet + result_bundle + mechanical judge`
  - 自动生成 `judgment_brief`
- 将 `literature_radar` 升级为真实可执行 loop：
  - 不再默认 `design_only`
  - 真实联网访问 OpenAlex
  - 真实写出 `literature_query_plan / literature_search_report / method_cards`
- 将 auto debate 从结构性 checklist 扩展为四角审查：
  - `Skeptic`
  - `Benchmark Steward`
  - `Literature Critic`
  - `Harness Reviewer`
- 将 `EXP-P1-002` 的人工批准写入 `human_review.json`
- daemon 自动将 `EXP-P1-002` 从 `awaiting_human_review` 释放为 `completed`
- proposer 自动 materialize `EXP-P1-003`
- 为 `EXP-P1-003` 自动补齐：
  - 默认 `output_dir`
  - 默认 `debate_bundle_path`
  - `task_plan.json`
  - `hypothesis_brief / design_pack / evaluation_rubric`
  - 四角 debate 产物
- 修复 release 后 completed card 的 stale acceptance task plan 污染

### 当前判断

- 当前系统已经不再只是“自动执行 harness”，而是具备了第一版真正的 `agentic autoresearch` 结构：
  - proposal
  - planning
  - debate
  - execution
  - analysis
  - context-aware judgment
  - human stop/release
- `EXP-P1-001` 与 `EXP-P1-002` 的 agentic judgment 现已与机械结论一致，且都保持 `promote`
- 当前主线已自动推进到 `EXP-P1-003`
- 当前真正的下一阻塞不再是基础设施，而是：
  - `learner_sensitivity_ladder` 尚无 runtime handler
  - 也即下一步需要把设计态 loop 编译成真正可执行实验协议

### 下一步建议

1. 将 `learner_sensitivity_ladder` 接成 executable loop
2. 让 `evaluation_rubric` 与 `design_pack` 直接参与该 loop 的 recipe 编译
3. 让 `EXP-P1-003` 完成后自动 materialize `EXP-P1-004`

## 2026-04-14（runtime synced）

### 本次动作

- 校对当前 runtime 真实状态，确认 detached supervisor 仍在运行
- 校对 `task_board.json / runtime_index.json / EXP-P1-003.json`，确认：
  - `EXP-P1-003` 已真实完成
  - judge 结果为 `promote`
  - `EXP-P1-004` 已自动 materialize 为下一条 planned card
- 同步修正 `.slicetune` 摘要层中仍停留在“`EXP-P1-003` 尚未执行”的陈旧状态

## 2026-04-14（EXP-P1-004 executable）

### 本次动作

- 为 `EXP-P1-004` 接通 `Tier A` executable runtime：
  - 新增 `helpers/trainability.py`
  - 在训练脚本中加入 `--trainable-modules`
  - 在 `feature_experiment_pipeline.py` 中接入 selective unfreeze / optimizer groups / train-mode control
- 新增 `research_harness/feature_intervention.py`
  - 读取 processed feature bundles
  - 计算 probe-axis score
  - 生成 `real_feature_guided` high/low subset manifests
  - 跑 learner-specific noise floor
  - 跑 learner x axis 的 Tier A pair runs
- 将 `feature_intervention_matrix` 接入：
  - `loop_catalog`
  - `preflight`
  - `run_research_tick.py`
  - `judge.py`
  - `task_progress.py`
  - `agentic_judge.py`
  - `analyst.py`
- 为新逻辑补充测试：
  - `test_research_trainability.py`
  - `test_research_feature_intervention.py`
- 通过测试：
  - `25 passed`
  - 后续针对 preflight 再跑一轮 `19 passed`
- 修复 `EXP-P1-004` 的 debate bundle 旧版本问题，并重新生成为可通过 gate 的四角 debate
- 发现旧 daemon 仍在使用旧版 loop catalog，于是重启 supervisor/daemon
- 发现 preflight 对 `clipdino2` 产生 CUDA false negative，实机确认不是机器无 GPU，而是 probe 导入顺序与 MKL 线程层冲突
- 修复 preflight probe 后，再次重启 daemon，并将 `EXP-P1-004` 重新入队
- 当前 `EXP-P1-004` 已在 `clipdino2` runtime 下进入 `running`
- 已写出：
  - `preflight_report.json`
  - `axis_scores_summary.json`
  - attempt manifest

### 当前判断

- `feature_intervention_matrix` 不再是设计态空壳，而是已接成真实可执行 loop
- 当前自治链路已实际完成：
  - debate gate
  - runtime profile selection
  - GPU-aware preflight
  - queue claim
  - daemon execution
- 当前 `EXP-P1-004` 尚未写出 `cache / manifests / runs / progress.json`
- 结合：
  - daemon 主进程持续高 CPU
  - `axis_scores_summary.json` 已落盘
  - 当前无 `session_error.json`
  最合理的解释是：当前正停留在 `class_presence` 的重前处理阶段，而不是已经失败

### 下一步建议

1. 持续观察 `EXP-P1-004` 是否开始写出 `cache / manifests / runs`
2. 一旦有首批 `noise floor` run 完成，优先确认：
   - `L0/L1/L2` 各自 noise floor 是否可用
   - realized target delta 是否足够大
3. 若 `class_presence` 前处理时间过长，再考虑为该步骤单独加 cache-progress 可视化，而不是立刻改实验逻辑

### 当前判断

- 当前系统已经越过“只有自动执行 harness”的阶段，进入“主线 loop 可以自动 materialize、自动 debate、自动执行、自动 judge、自动衔接下一张设计卡”的状态
- `EXP-P1-003` 给出的最重要结论是：
  - 当前最强可用 regime 仍是 `feature_experiment_fast_cached_slide`
  - learner regime 差异巨大，`regime_range=3.90`，远高于当前 training noise 与 global floor
  - Phase 1 下一步更适合进入 `feature_intervention_matrix`，而不是继续扩 learner 分支
- 当前框架层的真实主阻塞已从 `learner_sensitivity_ladder` 转移为 `feature_intervention_matrix` 缺少 runtime handler

### 下一步建议

1. 将 `feature_intervention_matrix` 接成 executable loop
2. 让其直接消费现有 `design_pack / evaluation_rubric / context_snapshot`
3. 跑通 `EXP-P1-004` 后，再决定是否需要继续 materialize `EXP-P1-005` 或触发 Literature Radar

## 2026-04-14（review refinement）

### 本次动作

- 复核当前 learner sensitivity 结论与实际训练代码
- 确认当前训练只优化：
  - `obj_proj`
  - `bkg_decoder`
- 确认以下大块均冻结：
  - `clip_backbone`
  - `FOUND`
  - `DINO`
- 重新审视 `EXP-P1-003` 的科学解释边界，判断其更接近 `protocol sensitivity audit` 而不是最终版 `learner algorithm sensitivity to data composition`

### 当前判断

- 用户提出的修正是关键的：
  - 当前更需要审计 learner 的可塑性与可更新范围
  - 而不是继续只比较 `epoch / config bundle`
- 因此下一步最合理的主线不应直接进入完整 `feature_intervention_matrix`
- 更合理的顺序是：
  1. 先建立 `learner adaptability audit`
  2. 比较不同 trainable scope / adaptation mechanism
  3. 再在最合适的 learner 上执行 `feature_intervention_matrix`

## 2026-04-14（P1-004 设计细化）

### 本次动作

- 将 `EXP-P1-004` 从泛化的 `feature_intervention_matrix` 设计，收紧为：
  - 最小 `learner adaptability audit`
  - 目标是比较不同 learner 可塑性是否放大真实 feature-guided intervention 的响应
- 新增实验细则文件：
  - `.slicetune/experiments/EXP-P1-004_design_spec.md`
- 冻结 3 个 learner variants：
  - `L0_head_only`
  - `L1_task_head_plus`
  - `L2_last_block_partial`
- 冻结 2 条主 probe axes：
  - `quality_sharpness`
  - `difficulty_small_object`
- 将 `coverage_density` 作为 only-after-promote 的 optional axis
- 冻结 3 类 controls：
  - `real_feature_guided`
  - `shuffled_feature_guided`
  - `matched_random_control`
- 冻结 4 个 reporting metrics：
  - `composition_response_amplitude`
  - `response_to_noise_ratio`
  - `directional_consistency`
  - `feature_validity_advantage`
- 将 active `hypothesis_brief / design_pack / evaluation_rubric` 从通用模板改写为面向 `EXP-P1-004` 的具体细则

### 当前判断

- 现有 feature space 当前应被视为 `working hypothesis`，不是 ground truth
- 当前最合理的实验台不是“大搜索”，而是：
  - 少量 learner variants
  - 少量 probe axes
  - 明确 controls
  - 每个 learner 单独估计 noise floor
- 当前最需要防止的误判是：
  - 把 protocol 混杂误判成 learner sensitivity
  - 把 intended feature shift 误判成真实 materialized intervention

### 下一步建议

1. 优先实现 `feature_intervention_matrix` 的 `Tier A` runtime handler
2. 在 runtime 中先接：
   - learner-specific noise floor
   - `quality_sharpness`
   - `difficulty_small_object`
   - `real_feature_guided`
3. 同步写出：
   - `realized_target_delta`
   - `off_target_drift`
   - `class_histogram_drift`
4. 待 `Tier A` 结果出现后，再决定是否推进 `Tier B`

## 2026-04-14（planner 专项修正）

### 本次动作

- 发现后台 daemon 会按 planner 通用模板重新生成 `EXP-P1-004` 的 `agentic` 产物
- 将 `research_harness/planner.py` 扩展为支持：
  - `design_mode=minimal_learner_adaptability_audit`
  - 从 card metadata 中编译：
    - learner variants
    - probe axes
    - control families
    - tier plan
    - metric definitions
    - 设计专属 judge contract
- 新增针对该分支的定向测试：
  - `tests/test_research_agentic.py`
- 使用更新后的 planner 重新生成了 `EXP-P1-004` 的 active agentic artifacts

### 当前判断

- `EXP-P1-004` 现在不再依赖手工改 artifact 才能保持细化设计
- 当前 design 层已经具备“card metadata -> agentic artifacts”的稳定编译路径
- 下一步的真实主阻塞已经回到 runtime：`feature_intervention_matrix` 还缺 executable handler

### 下一步建议

1. 以当前 `design_spec + design_pack + evaluation_rubric` 为合同，实现 `Tier A` runtime
2. 在实现中优先支持：
   - learner-specific noise floor
   - `L0/L1/L2`
   - `quality_sharpness / difficulty_small_object`
   - `real_feature_guided`

### 下一步建议

1. 设计最小 learner adaptability ladder：
   - `head-only`
   - `+ task head / decode head unfreeze`
   - `+ partial backbone unfreeze or PEFT`
2. 固定 `full` validation 与 matched data interventions
3. 评价指标改为：
   - composition response amplitude
   - signal-to-noise ratio
   - consistency across seeds

## 2026-04-14（design review）

### 本次动作

- 评审新的实验设计建议，重点审查：
  - 是否仍然保持当前 phase 的可解释性
  - 是否把 `learner` 与 `feature` 两个不确定变量同时放开过多
  - 是否能直接映射到当前仓库的 feature schema 与训练代码

### 当前判断

- 当前更合理的实验设计不是直接使用整个特征空间，而是：
  - 从现有 `quality / difficulty / coverage` 中挑少量 `probe feature axes`
  - 把现有特征当作 `working hypothesis`
  - 为每个 learner 单独估计 noise floor
  - 对每条 axis 做 `real / shuffled / random` 对照
- 但 `shuffled/random` 对照必须保持 matched materialization；否则会把 feature validity 和实例化难度混在一起
- 第一轮应优先控制规模，避免过早进入大 factorial search

### 下一步建议

1. 基于现有 processed schema 选 2 条最清晰的 probe axes
2. 选 2 到 3 个 learner variants 形成最小 ladder
3. 冻结正式指标：
   - `composition_response_amplitude`
   - `response_to_noise_ratio`
   - `directional_consistency`
   - `feature_validity_advantage`

## 2026-04-14（EXP-P1-004 runtime follow-up）

### 本次动作

- 读取 `EXP-P1-004` 第一条真实 worker 日志，确认首轮失败不是科学结果，而是 worker runtime 兼容问题
- 修复 `feature_experiment_pipeline.py` 在 Python 3.9 下的联合类型注解兼容问题
- 修复 `run_remix_training_experiment.py` 在 Hydra struct 模式下注入 `cfg.train.trainable_modules` 的失败
- 对修复做最小回归：
  - `clipdino2` 环境下导入 `feature_experiment_pipeline` 成功
  - `open_dict(cfg.train)` 注入 `trainable_modules` 成功
  - 定向测试 `8 passed`
- 将 `EXP-P1-004` 重新入队，并确认 daemon 在第 3 次 attempt 下重新进入 `running`

### 当前判断

- 当前还没有 `EXP-P1-004` 的科学结果；目前拿到的是“runtime 已真正进入 learner adaptability 训练链路”的执行证据
- 本轮已经确认：
  - processed feature 载入成功
  - axis scoring 成功
  - class-presence cache 构建成功
  - 四个 `real_feature_guided` manifests 已物化
  - `L0_head_only_noise_seed00` 已真正开始训练
- 最新 worker 日志已出现：
  - `starting training`
  - `[Trainability] modules=obj_proj,bkg_decoder trainable_params=4`

### 下一步建议

1. 等待首个 `L0_head_only_noise_seed00` 写出 `result.json + completion.json`
2. 若 `L0` noise-floor 成功完成，再继续观察 `L1/L2` selective-unfreeze 变体是否稳定进入训练
3. 在拿到首批 noise-floor 数值之前，不对 `EXP-P1-004` 做科学结论延伸

## 2026-04-14（EXP-P1-004 first result）

### 本次动作

- 持续监控 `EXP-P1-004` 的首条真实 noise-floor run
- 确认 `L0_head_only_noise_seed00` 完成训练与 full validation
- 读取：
  - `runs/L0_head_only_noise_seed00/result.json`
  - `runs/L0_head_only_noise_seed00/completion.json`
  - 更新后的 `task_plan.json / progress.md`

### 当前判断

- 当前已拿到 `EXP-P1-004` 的首个真实结果：
  - learner=`L0_head_only`
  - seed=`0`
  - `mIoU=24.29`
  - `mAcc=41.28`
  - `aAcc=38.6`
- 该数值与先前同 subset、同配置的 head-only 基线一致
- 因此当前最稳妥的解释是：
  - `feature_intervention_matrix` 新 runtime 没有明显破坏 baseline fidelity
  - 当前 selective-unfreeze 接线至少在 `L0` 路径上已经真实跑通
- 但这还不是 learner adaptability 的科学结论：
  - 只完成了一个 `L0` noise seed
  - 还没有 `L0` 完整 noise floor
  - 还没有 `L1/L2` 结果
  - 更没有 probe-axis pair 的 response 指标

### 下一步建议

1. 继续完成 `L0` 其余 noise seeds
2. 继续观察 `L1/L2` selective-unfreeze 是否稳定进入训练
3. 等 learner-specific noise floor 出来后，再解释：
   - `composition_response_amplitude`
   - `response_to_noise_ratio`
   - `directional_consistency`

## 2026-04-15（EXP-P1-004 Tier A 完成）

### 本次动作

- 修复 `judge_feature_intervention_matrix` 的 policy 参数兼容问题，避免 `design_mode` 触发 `TypeError`
- 将 `EXP-P1-004` 从旧的 `failed_execution` 状态重新入队，并在前台完成最终 judge/finalize
- 确认当前正式产物全部落盘：
  - `result_bundle.json`
  - `judge_report.json`
  - `agentic/analysis_brief.json`
  - `agentic/judgment_brief.json`
  - `noise_floor_summary.json`
  - `cell_results.json`

### 当前判断

- `EXP-P1-004` 当前正式状态为：
  - card=`completed`
  - judge=`park`
  - evidence=`E2`
- `Tier A` 的执行面已经完整跑通：
  - `3` 个 learner variants
  - `2` 条 probe axes
  - `6` 个 `real_feature_guided` learner-axis cells
- learner-specific noise floor 完全一致：
  - `L0=24.29`
  - `L1=24.29`
  - `L2=24.29`
  - `std≈0`
- 两条 probe axes 的 real response 在三个 learner 上也一致：
  - `quality_sharpness`：high=`23.73`，low=`23.88`，amplitude=`0.15`
  - `difficulty_small_object`：high=`24.15`，low=`24.28`，amplitude=`0.13`
- 当前更重要的科学结论不是“Tier A screen 通过”，而是：
  - 当前 `L0/L1/L2` 这组三档轻量 learner adaptability ladder 没有拉开数据组成敏感性的差异
  - 因此当前 selective-unfreeze 方案不足以证明“更大 trainable scope 会让 CLIP-DINOiser 更吸收 feature-guided composition signal”
- 同时，materialization fidelity 仍然是正面的：
  - `mean_off_target_drift_ratio=0.0154`
  - 说明当前 matched intervention 没有明显把大量 off-target drift 一起带偏

### 下一步建议

1. 若当前优先回答 feature 问题，则推进 `Tier B`：
   - 补 `shuffled_feature_guided`
   - 补 `matched_random_control`
   - 检查 `real` 是否真的优于 control
2. 若当前优先回答 learner 问题，则新开更强 learner 分支：
   - 不再默认沿当前 `L0/L1/L2` 轻量 ladder 继续细化
   - 考虑更强的 adaptation mechanism 或新的 learner family
3. 当前不应把 `EXP-P1-004` 解释成“learner adaptability 已建立”；它更接近：
   - `Tier A screen completed`
   - `current lightweight ladder did not differentiate learners`

## 2026-04-15（learner 梯度路径复盘）

### 本次动作

- 重新检查当前训练链路，重点核对：
  - `CLIP_DINOiser` 训练时真正用到的 feature 来源
  - `L1/L2` 新增 trainable modules 是否真的进入 loss 路径
  - 当前 head-only 与 selective-unfreeze 的差异是否有机会被 loss 感知

### 当前判断

- 当前默认配置 `feats_idx=-3`
- `CLIP_DINOiser` 在训练时从 `resblocks[-3].ln_2` 注册 hook，并把该输出 `detach()` 后写入 `train_feats['clip_inter']`
- `obj_proj / bkg_decoder` 的训练损失实际使用的是这个 `detach()` 后的 `clip_inter`
- 因此本轮 `L1/L2` 里新增的：
  - `clip_backbone.decode_head.proj`
  - `clip_backbone.backbone.visual.transformer.resblocks.-1`
  - `clip_backbone.backbone.visual.ln_post`
  很可能虽然被设成可训练，但并没有形成对当前 loss 的有效梯度贡献
- 这意味着：
  - `EXP-P1-004` 当前不能被简单解释为“放开更多参数也没用”
  - 更合理的解释是：“当前 learner ladder 设计可能没有真正测试到更深层 learner 可塑性”

### 下一步建议

1. 先做 gradient-flow audit：
   - 记录 `L0/L1/L2` 每个 trainable module 的 grad norm
   - 确认哪些模块实际收到非零梯度
2. 若要继续测 learner adaptability，优先修改训练链路而不是继续堆更多 unfreeze：
   - 要么改为使用 `final` features
   - 要么在审计模式下移除关键 hook 的 `detach()`
   - 要么把 hook 位置改到真正会被新增 trainable modules 影响的后层
3. 在做完梯度路径修正前，不应把当前 `L1/L2` 的负结果当成强科学反证

## 2026-04-15（真实 backbone-grad 链路接入）

### 本次动作

- 为 `DinoCLIP.get_clip_features` 新增显式 `track_grad` 支持
- 为 `MaskClip.extract_feat / forward` 新增显式 `track_grad` 支持
- 新增 audit 配置：
  - `feature_experiment_fast_cached_slide_backbone_grad`
  - 使用 `feats_idx=final`
  - 启用 `enable_clip_grad_for_training=true`
- 修复 `MaskClip.extract_v` 中在梯度路径下会破坏反向传播的原地加法
- 在训练循环中新增首步梯度摘要打印：
  - `[TrainabilityGrad] ...`

### 当前判断

- 当前已经不只是“理论上可训练”，而是有一条真正可用的 backbone-grad 实验链路
- 真实 smoke test 已确认下列模块收到非零梯度：
  - `obj_proj`
  - `bkg_decoder`
  - `clip_backbone.decode_head.proj`
  - `clip_backbone.backbone.visual.transformer.resblocks.-1`
  - `clip_backbone.backbone.visual.ln_post`
- 因此后续如果用这条新链路重跑 learner adaptability，才更接近真正的问题：
  - “让 CLIP backbone 进入 loss 后，数据组成敏感性会不会被放大？”

### 下一步建议

1. 先用 anchor subset 跑一条 `L2` backbone-grad smoke/validation run，确认训练曲线与 full validation 都稳定
2. 再决定是否将该配置接入：
   - 新的 learner adaptability audit card
   - 或 `EXP-P1-004` 的 follow-on stronger learner branch

## 2026-04-15（第一次 backbone-grad 手工 run 复盘）

### 本次动作

- 阅读用户手工运行的第一次 backbone-grad 终端输出与 `result.json`
- 对照 `[TrainabilityGrad]` 日志，核对新增 CLIP 模块是否真的收到梯度
- 定位当前训练链路中的门禁 bug，并修复

### 当前判断

- 这次手工 run 的 `mIoU=23.61` 不能直接被解释成“真正的 backbone-grad 实验结果”
- 最关键的证据不是 final metric，而是首步梯度日志：
  - `clip_backbone.decode_head.proj: params_with_grad=0/1`
  - `resblocks.-1: params_with_grad=0/12`
  - `ln_post: params_with_grad=0/2`
- 说明这次真实训练里，新增 CLIP 模块仍然没有进入有效梯度路径
- 根因是：
  - `set_train_mode_for_modules()` 先调用 `model.eval()`
  - 原实现用 `self.training` 作为 backbone-grad 开关
  - 导致真实训练时 `track_grad` 被错误关掉

### 已完成修复

- 将 `forward_pass()` 中的开关从：
  - `self.training and enable_clip_grad_for_training`
  改为：
  - `enable_clip_grad_for_training and torch.is_grad_enabled()`
- 并在与真实训练一致的 smoke test 下重新验证：
  - `obj_proj / bkg_decoder / decode_head.proj / resblocks.-1 / ln_post`
    全部收到非零梯度

### 下一步建议

1. 用修复后的代码重新手工跑同一条 anchor-subset 实验
2. 新一轮终端输出里，优先检查 `[TrainabilityGrad]` 是否显示新增 CLIP 模块不再是 `0` 梯度
3. 只有在这一点成立后，新的 `mIoU` 才能被解释为真正的 backbone-grad learner 结果

## 2026-04-15（第二次 backbone-grad 手工 run）

### 本次动作

- 阅读用户第二次手工运行的 backbone-grad 终端输出与：
  - `artifacts/manual_runs/backbone_grad_L2_anchor_seed0_rerun/result.json`
- 核对 `[TrainabilityGrad]` 是否显示新增 CLIP 模块真正收到梯度

### 当前判断

- 这次 run 已经是“真正有效的 backbone-grad learner run”，因为新增 CLIP 模块不再是 `0` 梯度：
  - `decode_head.proj: 1/1`
  - `resblocks.-1: 12/12`
  - `ln_post: 2/2`
- 当前结果为：
  - `mIoU=23.22`
  - `mAcc=41.16`
  - `aAcc=38.02`
- 与旧的 head-only anchor baseline `24.29` 相比，这次结果更低
- 但当前不能直接下“放开 backbone 会更差”的强结论，因为这次同时改了三件事：
  1. `feats_idx: -3 -> final`
  2. feature path 从 detached intermediate 改成了 gradient-tracked final
  3. learner 从 head-only 改成了 `L2 partial backbone`
- 因此当前最合理的解释是：
  - 真实 backbone-grad 路径已经技术上打通
  - 当前 `L2 + final-feature` 这个具体组合，在 anchor subset 上没有立刻带来收益
  - 但还不能据此判断“真正更可塑的 learner 一定无效”

### 下一步建议

1. 先在同一新配置下补一个 `L0`：
   - 仍用 `feature_experiment_fast_cached_slide_backbone_grad`
   - 但只训练 `obj_proj + bkg_decoder`
2. 再补一个同配置下的 `L1`
3. 只有这样，才能在同一 feature path / 同一 config 下比较：
   - `L0 vs L1 vs L2`
4. 在此之前，不把 `23.22` 当作“更强 learner 负结果”的最终证据

## 2026-04-15（manual learner runs 的 GPU 使用策略）

### 当前判断

- 当前服务器虽有 4 张 GPU，但现有 `run_remix_training_experiment.py` 训练路径还不是正确的 DDP 训练实现
- 具体原因：
  - 训练阶段没有把模型包成 DDP
  - `train_loader` 没有使用 `DistributedSampler`
- 因此当前最合理的 4 卡用法不是：
  - 单个实验直接 `--nproc_per_node=4`
- 而是：
  - 用 4 张卡并行跑 4 个单卡实验
  - 例如 `L0 / L1 / L2 / high-low manifest` 同时跑

### 下一步建议

1. 继续所有 manual learner audit 都保持单卡
2. 若要提速，优先做多实验并行调度
3. 真正的 4-GPU 单实验 DDP 要作为单独工程任务实现，不应和当前 Phase 1 科学实验混在一起

## 2026-04-15（intermediate-grad learner family 建立）

### 本次动作

- 响应用户提出的“保留中间特征，但解冻其上游 blocks”的新 learner 路线
- 新增配置：
  - `configs/feature_experiment_fast_cached_slide_intermediate_grad.yaml`
- 将手工命令生成器扩展为支持两条 family：
  - `final`
  - `intermediate`
- 定义 `intermediate-grad` 的最小 ladder：
  - `L0`: `obj_proj + bkg_decoder`
  - `L1`: `L0 + resblocks.-3`
  - `L2`: `L1 + resblocks.-4`

### 当前判断

- 这条新 family 与现有 `final-feature backbone-grad` 不应混在一起解释
- 它的核心问题不是“最后层能不能学”，而是：
  - 在仍然使用 `-3` 中间特征的前提下
  - 如果去掉 `detach`
  - 并解冻能影响该中间特征的 upstream blocks
  - learner 对数据组成是否会更敏感

### 梯度烟雾审计

- 已使用随机输入做单步反传验证：
  - `obj_proj: 2/2 grad`
  - `bkg_decoder: 2/2 grad`
  - `resblocks.-3: 8/12 grad`
  - `resblocks.-4: 12/12 grad`
- `resblocks.-3` 不是 `12/12` 并不意外：
  - hook 在 `resblocks[-3].ln_2`
  - hook 之后的 block 内参数不一定处于当前 loss 路径中
- 因此当前结论是：
  - `intermediate-grad` family 技术上成立
  - 可以继续跑 anchor baseline

### 下一步建议

1. 用 4 张 GPU 并行跑：
   - `L0 anchor seed0`
   - `L1 anchor seed0`
   - `L2 anchor seed0`
   - `L2 anchor seed1`
2. 先建立这条新 family 的 baseline / 初步 noise 感知
3. 然后再决定是否在该 family 下跑 `quality_sharpness / difficulty_small_object` high-low

## 2026-04-15（intermediate-grad anchor 结果）

### 本次动作

- 读取用户完成的 4 个 `intermediate-grad` 手工 runs：
  - `intermediate_grad_L0_anchor_seed0`
  - `intermediate_grad_L1_anchor_seed0`
  - `intermediate_grad_L2_anchor_seed0`
  - `intermediate_grad_L2_anchor_seed1`

### 结果摘要

- `L0 anchor seed0`: `mIoU=24.29`
- `L1 anchor seed0`: `mIoU=24.20`
- `L2 anchor seed0`: `mIoU=23.52`
- `L2 anchor seed1`: `mIoU=23.52`

### 当前判断

- 这轮结果说明：
  - `intermediate-grad` family 技术链路有效
  - 但随着 trainable scope 从 `L0 -> L2` 增大，anchor baseline 没有变好，反而下降
  - `L2` 的两个 seed 几乎完全重合，至少在 anchor baseline 上没有呈现“更敏感但更噪”的明显模式
- 这仍然不能直接回答 feature sensitivity，因为目前只跑了 anchor baseline，没有跑 high/low interventions
- 但它已经提供了一个很强的研究信号：
  - 当前继续围绕 CLIP-DINOiser 的 teacher-distillation learner family 深挖，边际价值可能正在下降

### 当前建议

1. 不把当前结果解释成“feature 无效”
2. 也不把它简单解释成“更可塑 learner 无效”
3. 更合理的下一步是认真考虑引入一个更标准、更 data-sensitive 的 supervised segmentation learner 作为 `probe learner`
4. 用该 probe learner 先回答：
   - feature-defined data composition 是否真的能稳定影响真实 segmentation 训练结果
5. 然后再决定是否回到 CLIP-DINOiser / slice-remix 主线做绑定

## 2026-04-15（诊断型基础 learner 方案 review）

### 本次动作

- review 另一条建议提出的：
  - `frozen dense feature + linear/MLP probe`
  - small segmentation network
  - nearest-centroid / logistic sanity check
- 对照当前本地仓库与环境，判断其研究价值与接入成本

### 当前判断

- 这条建议的核心方向是对的：
  - 当前不应继续优先在 `CLIP-DINOiser` 上解冻更多层
  - 应引入诊断型基础 learner
- 但它里面混了两类不同作用的 learner：
  1. representation-level probe
  2. real supervised image-space probe

### 结论

- `frozen dense feature probe (P0/P1)` 值得做，但更适合当：
  - 便宜 sanity test
  - feature validity 预筛
- 它不应单独承担主诊断角色，因为它更回答：
  - frozen representation 中是否已有可分性 / feature signal
  而不是：
  - real segmentation learner 对数据组成是否敏感
- 当前更适合作为主 probe learner 的是：
  - `DeepLabV3+ R50-D8`
- 原因：
  - 环境内已现成安装 MMSeg 0.27
  - 本地已具备官方 config 库
  - 它是成熟、稳定、容易解释的 supervised segmentation baseline
- `SegFormer MiT-B0` 适合作为第二阶段交叉验证，不是第一优先级

### 当前建议

1. 研究顺序采用两级：
   - Level A: `frozen dense feature probe (P0/P1)`
   - Level B: `DeepLabV3+ R50-D8`
2. 如果只能先做一个主 probe learner，优先 `DeepLabV3+ R50-D8`
3. 等主 probe learner 回答了 feature sensitivity 是否成立，再决定是否引入 `SegFormer MiT-B0`

## 2026-04-15（DeepLabV3+ R50-D8 主 probe learner 接入）

### 本次动作

- 新增 `DeepLabV3+ R50-D8` 的手动 supervised probe 训练入口：
  - `research_harness/supervised_probe.py`
  - `run_supervised_probe_experiment.py`
- 当前实现能力：
  - 从 subset manifest 读取训练子集
  - 构建 MMSeg `DeepLabV3+ R50-D8` 配置
  - 将 COCO-Stuff train split 过滤到 manifest basenames
  - 单卡 supervised 训练
  - full COCO-Stuff val 评测
  - 写出统一 `result.json`

### 轻量验证

- CLI `--help` 已通过
- dataset subset smoke 已通过：
  - anchor manifest `rand_subset_s0145_t00`
  - 过滤后的训练集长度为 `1000`

### 当前判断

- 这条主 probe learner 已经具备“手工可跑”的最低条件
- 当前最合适的下一步不是继续扩脚本，而是先跑第一条 anchor 命令看真实效果

## 2026-04-15（DeepLabV3+ 单卡 `SyncBN` 故障修复）

### 现象

- 用户第一次手工运行 `DeepLabV3+ R50-D8` anchor probe 时，模型与权重初始化完成，但在第一步训练 forward 时报错：
  - `RuntimeError: Default process group has not been initialized`

### 原因

- 官方 MMSeg `DeepLabV3+ R50-D8` config 默认使用 `SyncBN`
- 当前 `run_supervised_probe_experiment.py` 是单卡、非分布式训练脚本，没有初始化默认 distributed process group
- 因此 `SyncBN` 会在真实训练时必然失败

### 修复

- 在 `research_harness/supervised_probe.py` 中加入递归配置改写：
  - 将 `cfg.model` 中全部 `SyncBN` 统一替换为普通 `BN`

### 验证

- 修复后的 `max_iters=1` smoke 已确认：
  - 成功进入真实训练迭代
  - 成功保存 checkpoint
  - 成功进入 full validation

### 当前意义

- 这次失败不是主 learner 路线错误，而是单卡运行时配置不兼容
- 现在 `DeepLabV3+ R50-D8` 的单卡 supervised probe 链路已经真实打通
- 下一步应重新运行正式 anchor 命令，读取 `result.json` 再判断：
  - `DeepLabV3+` 是否值得固定为 Phase 1 主 probe learner
  - 还是需要进一步压缩训练预算 / 调整 crop size / batch size

## 2026-04-15（DeepLabV3+ 首次 anchor 正式结果）

### 结果

- `artifacts/manual_runs/deeplabv3plus_r50_d8_anchor_seed0/result.json`
- 关键指标：
  - `mIoU=0.03`
  - `mAcc=0.60`
  - `aAcc=0.70`

### 训练状态判断

- 这次不是链路故障，训练完整跑完了
- 但训练日志显示：
  - loss 约从 `4.48` 降到 `3.77`
  - 说明模型在学，但明显远未收敛

### 当前解释

- 这次结果不应被解释为：
  - feature 没有 signal
  - `DeepLabV3+` 不适合作为 probe learner
- 更合理的解释是：
  - 当前 supervised probe 的训练 budget 还不成立
  - `1000` images, batch `2`, `1000` iters 只相当于约 `2` 个 epoch
  - 对随机初始化 segmentation heads 的 `DeepLabV3+` 来说，这个预算远不足以形成可用 mIoU

### 当前结论

- `DeepLabV3+ R50-D8` 主 probe learner 路线：`keep`
- 这次 `1000 iter` anchor 结果：`invalid as scientific evidence`
- 下一步应优先：
  - 增大 `max_iters`
  - 必要时重新校准学习率
  - 先把 supervised probe budget 跑到“至少能形成正常 anchor baseline”的程度，再进入 feature high/low 对比

## 2026-04-15（表中 open-vocabulary baselines 适配性筛选）

### 目标

- 用户希望参考 `CLIP-DINOiser` 论文中的历史方法，判断哪些值得在当前仓库中试跑

### 筛选标准

1. 官方代码是否明确可用
2. 与当前仓库的依赖/代码血缘是否接近
3. 是否适合作为当前 Phase 1 的辅助 baseline，而不是把主线切回复杂 open-vocabulary 预训练

### 当前 shortlist

- 第一优先级：`CLIP-DIY`
- 第二优先级：`MaskCLIP / MaskCLIP+`
- 第三优先级：`TCL`
- 第四优先级：`GroupViT`

### 当前不建议作为首波接入

- `ReCo`
- `NamedMask`
- `OVDiff`
- `OVSegmentor`
- `SegCLIP`
- `ZeroSeg`
- `CLIPpy`

### 原因摘要

- 当前仓库已经明确有：
  - `MaskCLIP` 血缘
  - `TCL` 血缘
  - `GroupViT` dataset/config 血缘
  - `FOUND` 安装与调用
- 因此：
  - `CLIP-DIY`：同作者方法、training-free、FOUND 依赖已满足，最值得先试
  - `MaskCLIP / MaskCLIP+`：工程血缘最近，最容易快速复用
  - `TCL / GroupViT`：有官方代码，但完整训练/环境更重，适合第二梯队
- 其余方法多数代码虽存在，但需要更重的数据准备、专用环境或额外生成/检索流水线，不适合当前第一波执行

## 2026-04-15（DeepLab 低结果的重新解释：不是任务结论，而是 protocol bug）

### 新现象

- 用户继续完成了第二轮 `DeepLabV3+ R50-D8` anchor：
  - `8000 iter`
  - 结果仍极低：`mIoU=0.08`
- 这比第一次 `1000 iter` 的 `0.03` 只略高一点，远低于可解释 supervised baseline

### 新判断

- 这已经不是“COCO-Stuff 动态范围小”可以解释的现象
- 结合进一步排查，更像 supervised probe 接入仍存在 protocol 层错误

### 定位结果

- 当前 `research_harness/supervised_probe.py` 借用了官方 `ADE20K` 训练 pipeline
- 该 pipeline 的 `LoadAnnotations` 默认：
  - `reduce_zero_label=True`
- 但本项目使用的是 `COCO-Stuff164k`
  - train IDs 为 `0..170`
  - `255` 才是 ignore
- 因此沿用 `reduce_zero_label=True` 会系统性破坏标签监督

### 已修复

- 在 `build_supervised_probe_cfg()` 中显式改为：
  - `LoadAnnotations(reduce_zero_label=False)`

### 当前研究结论

- 现在还不能因为 `DeepLab` 的低结果就得出：
  - `COCO-Stuff` 不适合
  - `image segmentation` 不适合
- 更合理的结论是：
  - supervised probe 的协议还在收口中
  - 需要先在修复后的 pipeline 上重新建立有效 anchor baseline

### 任务/数据集层面的判断

- 结合 `CLIP-DINOiser` 论文表格本身：
  - `COCO-Stuff` 列并非没有算法动态范围
  - `City`、`ADE` 的方法差异更大
- 所以如果后续真的要调整研究对象，更合理的顺序应是：
  1. 先修好 supervised probe 协议
  2. 再判断 `COCO-Stuff` 是否过高熵
  3. 若要降级，优先换到更低熵 segmentation benchmark，而不是立刻换任务类型

## 2026-04-15（修复后 DeepLab=1.11：开始指向 benchmark 过高熵，而不是任务失败）

### 新结果

- 修复 `reduce_zero_label=False` 后，用户重新跑完：
  - `DeepLabV3+ R50-D8`
  - `8000 iter`
  - anchor subset
- 新 full-val 结果：
  - `mIoU=1.11`
  - `mAcc=2.64`
  - `aAcc=22.79`

### 训练面判断

- 训练日志明显比旧版本健康：
  - `decode.acc_seg` 常见于 `18%~26%`
  - `decode.loss_ce` 约在 `2.2~2.6`
- 这说明：
  - 当前 supervised probe 已经不再是“标签协议损坏导致完全学不动”

### Full-val 失败形态

- per-class 结果显示：
  - 大约 `150 / 171` 类 IoU 仍为 `0`
  - 仅少量大类 / stuff 类有明显响应：
    - `sky-other`
    - `grass`
    - `snow`
    - `tree`
    - `wall-concrete`
    - `sea`
    - `person`
    - `road`

### 当前解释

- 这不再只是工程 bug
- 但也还不足以推导“segmentation 任务不行”
- 更像是：
  - `COCO-Stuff-171`
  - `1000` 张随机训练子集
  - full validation
  - `ImageNet` 预训练 backbone + 随机初始化 segmentation heads
  这个组合对当前 diagnostic learner 过于高熵

### 当前建议

- 不换任务类型
- 先保留 `image segmentation`
- 将 benchmark 难度下调，优先尝试：
  1. `COCO-Object`
  2. `VOC20`
  3. `Pascal Context59`

### 理由

- 这样仍保留：
  - 当前仓库的主要代码资产
  - 与 `CLIP-DINOiser` 系列的可比性
  - 已有 feature / slice / remix 资产
- 同时能更快判断：
  - feature-defined data composition 是否会影响真实 segmentation learner
- 本地进一步复核：
  - `VOC20` 代码入口已现成存在：
    - `segmentation/datasets/pascal_voc20.py`
    - `segmentation/configs/_base_/datasets/pascal_voc12_20.py`
  - 但 `data/VOCdevkit/VOC2012` 当前尚未准备好
  - 因此 `VOC20` 是一个科学上干净、工程上可做的切换候选，但不是零成本秒切

## 2026-04-15（为什么会出现 `150/171` 类 IoU 为 `0`）

### 排除掉的解释

- 训练子集缺类并不是主因
- 对当前 `1000` 图 anchor subset 统计后：
  - 实际覆盖了 `170 / 171` 类
  - 只缺 `waterdrops`

### 真正的问题

- 这个子集的类分布极端长尾：
  - `132` 个类只出现在 `<=50` 张图
  - `150` 个类只出现在 `<=100` 张图
- 因此对 `COCO-Stuff-171 full-val` 来说：
  - 模型即使学会了一部分常见类
  - 也会因为大量尾部类全零而被 `mIoU` 极度压低

### 与当前结果的关系

- 修复后：
  - `aAcc=22.79`
  - `mIoU=1.11`
- 这两者的巨大差距说明：
  - 模型不是“完全随机”
  - 它在常见像素上已经学到了一些模式
  - 只是等权平均到 `171` 个类后，被长尾类全部拖垮

### 额外诊断

- 当前结果里共有约 `21` 个类 IoU 非零
- 若只对非零类取均值，约为 `9.05`
- 这仍然不高，但比 `1.11` 更真实地反映了训练状态

### 当前研究含义

- 这更像 benchmark / metric 设计对当前 diagnostic learner 不友好
- 不应直接解读为：
  - `segmentation` 任务失败
  - 或 `COCO-Stuff` 毫无信号

## 2026-04-16（文献方向扩展：从 slice 扩到 data quality / curation / benchmark design）

### 当前问题重写

- 当前不再只问：
  - `slice` 是否能 debug model
- 还要问：
  - 能否基于样本特征或模型导出信号提升数据质量
  - 能否构造更适合当前场景的数据集
  - 哪类 benchmark / task 更容易让数据组成效应显性化

### 当前最相关的四条文献线

- `training dynamics / data maps`
  - 代表：`Dataset Cartography`
- `label quality / label error detection`
  - 代表：`Confident Learning`、segmentation label-quality work
- `model-aware subset selection / data valuation / data curation`
  - 代表：`GLISTER / GRAD-MATCH / CRAIG / Data Shapley / JEST`
- `distribution-shift / long-tail / multi-domain benchmark`
  - 代表：`WILDS / BREEDS / MESS / AUCSeg / DataPerf / DataComp`

### 对 SliceTune 的直接启发

- 后续不应只依赖当前手工 secondary features
- 更合理的是把：
  - `interpretable data features`
  - 与 `model-derived signals`
  结合成双源数据画像
- 若当前 benchmark 对数据组成不敏感，应优先考虑：
  - `long-tail`
  - `subpopulation shift`
  - `multi-domain`
  更显性的 segmentation setting，而不是立刻放弃任务类型

## 2026-04-16（是否可以彻底换 learner + dataset）

### 当前判断

- 可以
- 但不应该“任意换”

### 当前最有希望的成功区间

- `moderate-capacity supervised learner`
- `small/medium training budget`
- `low-entropy + structured-shift segmentation benchmark`

### 为什么不是超强 foundation model

- Phase 1 当前目标是放大数据组成效应
- 超强 foundation segmentation model 很可能再次把数据效应洗平
- 当前更需要的是：
  - 会真实吸收训练数据差异
  - 但又不会强到把 benchmark 做平的 learner

### 对 benchmark 的含义

- `VOC20` 这类低熵 benchmark 更适合先拿到稳定 baseline
- 但若要让 current features 真正显性化，后续更可能需要：
  - `long-tail`
  - `subpopulation shift`
  - `multi-domain`
  的 segmentation setting

## 2026-04-16（外部建议 review：diagnostic benchmark–learner pair）

### 当前认可的部分

- “先找第一个能稳定放大数据组成效应的 `diagnostic benchmark–learner pair`”这个总目标是对的
- `CLIP-DINOiser` 不再适合继续承担 Phase 1 主诊断 learner
- 继续使用：
  - direct feature-guided intervention
  - `real / shuffled / matched-random`
  - local metrics
  的思路是对的
- `DeepLabV3+` 主 learner、`SegFormer-B0` challenger 的高层路线合理

### 当前需要修正的部分

- `COCO-Object` 虽然是最低迁移成本候选，但本地 `_instanceTrainIds.png` 标注尚未转换生成
- 当前 supervised probe 代码仍然硬编码到 `COCO-Stuff`
- `Cityscapes` 本地尚未完全展开到可直接训练状态，`ACDC` 当前也未准备

### 当前更稳的执行顺序

1. 先把 supervised probe 变成 dataset-aware runner
2. 先运行 `tools/convert_coco_object.py`，再做 `COCO-Object`
3. 若 signal 仍弱，再切 `Cityscapes/ACDC`
4. `VOC20` 保留为最干净的 sanity benchmark

### 2026-04-16 更正

- 仓库里确实已经有：
  - `tools/convert_coco_stuff.py`
  - `tools/convert_coco_object.py`
- README 里旧写法 `tools/convert_coco.py` 已过期
- `convert_coco_object.py` 的真实逻辑是：
  - 直接读取当前 `data/coco_stuff164k/annotations/{train2017,val2017}` 下的 raw `*.png`
  - 跳过已生成的 `*TrainIds.png`
  - 将 `<=90` 的 COCO thing 类 remap 为 object labels
  - 将 `>90` 的 stuff 类整体置为 `background=0`
  - 最终输出 `*_instanceTrainIds.png`
- 因此：
  - 生成 `COCO-Object` labels 不需要额外的 COCO instance JSON
  - `COCO-Object` 的工程门槛低于我们前一轮判断

## 2026-04-16 DeepLabV3+ probe 解释澄清

- 当前 `DeepLabV3+` 这条线已经需要和“原论文复现”明确切开：
  - 本地 probe 入口 [research_harness/supervised_probe.py](/home/yuhe/clip_dinoiser/research_harness/supervised_probe.py:1) 借用的是 MMSeg 的 `deeplabv3plus_r50-d8_512x512_80k_ade20k.py`
  - 然后手工改成：
    - `COCOStuffDataset`
    - `171` 类
    - `1000` 图 manifest 子集
    - full COCO-Stuff validation
- 这意味着当前实验是：
  - “借用标准 supervised segmentation 架构的本地 diagnostic probe”
  - 而不是对原始 DeepLabV3+ 论文 benchmark recipe 的 faithful reproduction
- 最新修复标签协议后的有效结果：
  - `mIoU=1.11`
  - `mAcc=2.64`
  - `aAcc=22.79`
  - `21` 个类别 IoU 非零
  - 非零类别平均 IoU 约 `9.05`
- 解释：
  - 当前训练已不是随机或完全坏掉
  - 但在 `COCO-Stuff-171 + 1000 图随机子集 + full-val` 组合下，长尾被压得过重
  - 因而这条结果更应解释为：
    - `benchmark–learner–metric` 契约过苛
    - 而不是 `DeepLabV3+` 架构本身无效

## 2026-04-16 COCO-Object 数据就绪性再更正

- 本地独立目录 [data/coco_object](/home/yuhe/clip_dinoiser/data/coco_object) 已存在且内容基本齐全
- 因此：
  - `COCO-Object` 不必再被视为必须先做标注转换的前置 gate
  - 当前主工程任务应转为：
    1. 让 supervised probe 变成 dataset-aware runner
    2. 直接支持 `COCOObjectDataset`
    3. 在 `COCO-Object` 上重新建立 diagnostic benchmark–learner pair

## 2026-04-16 benchmark 解释进一步澄清

- `VOC20` 与 `Cityscapes` 都比 `COCO-Stuff-171` 更适合作为下一阶段 supervised probe benchmark，但原因不同：
  - `VOC20` 更适合做最干净的 sanity benchmark
  - `Cityscapes` 更适合寻找结构化分布效应
- 这不是简单的“哪个更简单”：
  - `VOC20`：
    - 类别数低
    - object-centric
    - 更适合先验证 learner 是否能稳定学起来
  - `Cityscapes`：
    - 类别熵低
    - 场景结构高度规则
    - 对天气/夜间/能见度/小目标相关 slice 更自然
- 与此同时，`CLIP-DINOiser` 和 `DeepLabV3+` 的论文数字不能直接按同一规则比较：
  - `DeepLabV3+` 原论文报告的是 fully supervised closed-set segmentation benchmark 结果
  - `CLIP-DINOiser` 论文表中的行属于 open-vocabulary / frozen-CLIP family setting
- 因而：
  - `CLIP-DINOiser` 低于 `DeepLabV3+` 的论文数字，不能直接解释成算法退化
  - 更合理的解释是：
    - 监督设定不同
    - 任务难度不同
    - 论文数字代表的是不同问题定义下的性能

## 2026-04-16 开放词汇路线的阶段定位

- 当前进一步明确：
  - 开放词汇 / frozen-foundation segmentation 并不是不值得做
  - 它只是当前不适合继续承担 Phase 1 的第一性信号验证职责
- 原因：
  - 若直接在 open-vocabulary setting 上成功，论文主张会更强
  - 但在当前阶段，它的失败解释空间太大：
    - feature 无效
    - benchmark 无效
    - learner 太强先验导致对数据组成不敏感
    - 弱监督 / distillation 路径把局部 signal 洗平
- 因而当前更稳的研究顺序是：
  1. 先在 supervised diagnostic benchmark–learner pair 上证明数据组成效应存在
  2. 再回接到 `CLIP-DINOiser` 或其他 open-vocabulary family
  3. 若回接成功，再把“对更难开放词汇任务也成立”作为更强主张

## 2026-04-17 工作区重构与官方 DeepLab 接入

- 已创建统一工作区根目录：
  - `/home/yuhe/slicetune`
- 已将主研究仓库移动到：
  - `/home/yuhe/slicetune/clip_dinoiser`
- 为兼容当前 `.slicetune` 历史绝对路径与 artifact，保留软链接：
  - `/home/yuhe/clip_dinoiser -> /home/yuhe/slicetune/clip_dinoiser`
- 已将官方 DeepLab 代码以并列目录方式接入：
  - `/home/yuhe/slicetune/deeplab`
- 接入方式：
  - 从官方仓库 `https://github.com/tensorflow/models.git` 进行 sparse clone
  - 当前只展开 `research/deeplab` 工作树
- 当前意义：
  - 后续可以在同一工作区下直接对照 `clip_dinoiser` 与官方 DeepLab 代码
  - 同时不破坏现有绝对路径引用与会话环境

## 2026-04-17 官方 DeepLab 复现流程审阅

- 已阅读官方 `DeepLab` repo 中与复现最相关的文件：
  - `README.md`
  - `g3doc/installation.md`
  - `g3doc/model_zoo.md`
  - `g3doc/pascal.md`
  - `g3doc/cityscapes.md`
  - `local_test.sh`
  - `train.py / eval.py / model.py / datasets/data_generator.py`
- 当前确认：
  - 官方“复现论文结果”需要一个 benchmark-specific 的完整流程，而不是单独跑 `train.py`
  - 其标准操作链包括：
    1. 安装 TensorFlow 与依赖
    2. 配置 `PYTHONPATH`
    3. 将原始数据转换成 `TFRecord`
    4. 选择初始 checkpoint 或作者发布 checkpoint
    5. 训练
    6. 评测
    7. 可视化 / 导出
- 这进一步支持当前判断：
  - 我们之前在 MMSeg 中借用 `DeepLabV3+` 架构得到的 probe 结果，不应被解释成官方 TensorFlow DeepLab repo 的直接 reproduction

## 2026-04-17 官方 DeepLab 环境兼容性修复

- 用户在 `deeplab` conda 环境中运行：
  - `python deeplab/model_test.py`
  首先报错：
  - `Descriptors cannot not be created directly`
- 检查环境版本后确认：
  - `tensorflow-gpu==1.15.5`
  - `protobuf==4.24.4`
  - 属于典型 `TF1.x + protobuf 4.x` 不兼容
- 已执行修复：
  - 将 `protobuf` 降级为 `3.20.3`
- 修复后暴露第二层问题：
  - `ModuleNotFoundError: No module named 'nets'`
- 原因确认：
  - 当前官方仓库只 sparse checkout 了 `research/deeplab`
  - 但 `local_test.sh` 与实际导入路径都要求：
    - `research`
    - `research/slim`
    同时存在于工作树与 `PYTHONPATH`
- 已执行修复：
  - 为官方 repo 展开 `research/slim`
- 修复后暴露第三层问题：
  - `ModuleNotFoundError: No module named 'tf_slim'`
- 已执行修复：
  - 在 `deeplab` 环境中安装 `tf_slim`
- 最终结果：
  - `deeplab/model_test.py` 成功通过：
    - `Ran 5 tests in 8.709s`
    - `OK (skipped=1)`
- 当前意义：
  - 官方 TensorFlow DeepLab 代码链路已具备最小可运行性
  - 当前可将后续重点从“修 import / 修依赖”切换到：
    - benchmark 数据准备
    - `local_test.sh` / `pascal.md` 的最小端到端验证

## 2026-04-17 官方 DeepLab `local_test.sh` 数据下载失败

- 在 `model_test.py` 已通过的前提下，继续运行 `local_test.sh`
- 当前首个失败点不是训练，而是：
  - `download_and_convert_voc2012.sh`
  - 下载 `https://data.deepai.org/PascalVOC2012.zip`
- 当前服务器报错：
  - IPv4 连接超时
  - IPv6 不可达
- 当前判断：
  - 这是旧镜像地址的可达性问题
  - 不是官方 DeepLab TensorFlow 训练/评测代码本身失效
- 当前行动建议：
  - 不再把 `local_test.sh` 当作完全黑箱一键入口
  - 改为：
    1. 手工准备 `VOC2012` 原始数据
    2. 单独运行 `remove_gt_colormap.py`
    3. 单独运行 `build_voc2012_data.py`
    4. 再回到 `train.py / eval.py`

## 2026-04-17 官方 DeepLab VOC 下载脚本源切换

- 已直接修改官方脚本：
  - `deeplab/datasets/download_and_convert_voc2012.sh`
- 当前修改内容：
  - 将下载源从：
    - `https://data.deepai.org/PascalVOC2012.zip`
    切换到：
    - `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar`
  - 将解压方式从：
    - `unzip`
    切换为：
    - `tar -xf`
  - 将数据根路径从：
    - `pascal_voc_seg/VOC2012`
    修正为：
    - `pascal_voc_seg/VOCdevkit/VOC2012`
- 当前判断：
  - 这属于“修下载入口”，不属于“修改 DeepLab 训练协议”
  - 因为后续 `build_voc2012_data.py` 真正依赖的就是官方 VOC tar 解压后的目录结构

## 2026-04-17 官方 DeepLab VOC 数据准备完成

- 当前已确认 `download_and_convert_voc2012.sh` 跑完后，`PASCAL VOC 2012` 数据准备完整
- 已生成：
  - 原始图像目录：`VOCdevkit/VOC2012/JPEGImages`
  - 原始标签目录：`VOCdevkit/VOC2012/SegmentationClass`
  - 去 colormap 后标签目录：`VOCdevkit/VOC2012/SegmentationClassRaw`
  - split 列表：`train.txt / val.txt / trainval.txt`
  - `TFRecord`：
    - `train-00000..00003-of-00004`
    - `val-00000..00003-of-00004`
    - `trainval-00000..00003-of-00004`
- 当前统计：
  - `JPEGImages`: `17125`
  - `SegmentationClass`: `2913`
  - `SegmentationClassRaw`: `2913`
- 当前阶段判断：
  - 数据准备 gate 已通过
  - 后续应将重点切换到：
    1. 初始 checkpoint 准备
    2. 最小 `train.py / eval.py`

## 2026-04-17 官方 DeepLab `local_test.sh` 成功通过

- 当前 `local_test.sh` 的最小 `PASCAL VOC` train/eval 链路已成功运行
- 关键评测结果：
  - `eval/miou_1.0_overall = 0.821973264`
- 与脚本内预期值基本一致：
  - `mIOU=82.20%`
- 当前解释：
  - 这已经足够证明：
    - 官方 checkpoint 可用
    - 当前 TensorFlow 1.x DeepLab 环境可用
    - 当前 `PASCAL VOC` 数据准备与 `TFRecord` 转换可用
    - 最小 `train.py / eval.py` 逻辑在本机上可跑通
- 当前边界：
  - 这仍属于官方提供 checkpoint 基础上的最小演示链路
  - 还不等于完整“从头训练复现论文主结果”

## 2026-04-17 supervised segmentation 数据特征文献整理

- 当前围绕“哪些数据特征会显著影响 supervised image segmentation 训练效果”做了第一轮文献归纳
- 当前最稳定的结论不是某单一 feature，而是 5 类 recurring factors：
  1. 标注质量 / annotation noise / boundary ambiguity
  2. 类别长尾与像素不平衡
  3. 小目标比例与对象尺度
  4. 图像质量与 adverse conditions
  5. 域组成 / 分布偏移
- 当前启发：
  - 若后续要为 SliceTune 寻找更“feature-sensitive”的 benchmark–learner pair，最值得优先探索的不是抽象 clustering，而是这些已有文献支持的数据轴

## 2026-04-17 supervised learner feature-sensitivity metric contract

- 当前将“learner 是否对某 feature 敏感”的判断，固定为四个量，而不是只盯 `mIoU`
- 当前定义：
  - `response_amplitude`:
    - `Δ_real = metric(high) - metric(low)`
  - `response_to_noise_ratio`:
    - `RNR = |Δ_real| / sigma_noise`
  - `control_gap`:
    - `CG = |Δ_real| - max(|Δ_shuffled|, |Δ_random|)`
  - `directional_consistency`:
    - 多 seed 下 `Δ_real` 的方向是否一致
- 当前解释：
  - 真正的“敏感”，不是单次 high/low 有差
  - 而是：
    1. 这个差值大于 seed noise
    2. 这个差值大于 shuffled/random control
    3. 这个差值方向稳定
- 当前 metric 备注：
  - `global mIoU` 仍保留
  - 但对局部 feature 必须增加局部指标：
    - `small-object mIoU`
    - `rare-class mIoU`
    - `present-class mIoU`

## 2026-04-17 新 benchmark feature extraction 迁移策略

- 当前对“切 benchmark 后是否要重写特征提取代码”的判断是：
  - **不应重写整套特征算法**
  - 应改为：
    1. 复用 benchmark-agnostic extractor
    2. 新写 dataset adapter
    3. 对 label-derived features 做 dataset-aware 参数化
- 当前已确认可直接复用的层：
  - `quality` 提取入口 [extract_quality_raw_features.py](/home/yuhe/slicetune/clip_dinoiser/extract_quality_raw_features.py:47)
    - 只要求 `subset_root + sample_index.npy`
  - `coverage` embedding 入口 [extract_coverage_embeddings.py](/home/yuhe/slicetune/clip_dinoiser/extract_coverage_embeddings.py:47)
    - 只要求 `image_rel -> image_path`
- 当前已确认必须随 benchmark 适配的层：
  - `sample_index` / `image_rel` / `annotation_rel` 契约
  - `class_presence` 路径和 label 协议
  - `small_object_ratio` 中：
    - `thing_id_start`
    - `num_things`
    - `ignore_index`
  - 任何依赖具体标签语义的 rare-class / present-class 统计
- 当前推荐的 feature verification ladder：
  1. schema / shape / range / empty-case sanity
  2. 手工 visual spot-check
  3. 独立脚本交叉重算
  4. 分布 sanity
  5. high/low materialization face-validity 检查
- 当前解释：
  - 这条迁移策略能避免每换一个 benchmark 就重写所有 extractor
  - 同时又能让我们对“特征是否真的按预期计算”保有足够强的验证证据

## 2026-04-17 `VOC20` feature-adapter 第一版已落地

- 当前已落地的适配层包括：
  - `dataset spec` registry：
    - [dataset_specs.py](/home/yuhe/slicetune/clip_dinoiser/feature_utils/data_feature/dataset_specs.py:1)
  - `VOC20` subset builder：
    - [sample_voc20_subset.py](/home/yuhe/slicetune/clip_dinoiser/tools/sample_voc20_subset.py:1)
  - generalized `class_presence`:
    - [class_coverage.py](/home/yuhe/slicetune/clip_dinoiser/slice_remix/class_coverage.py:10)
- 当前关键实现点：
  - `VOC20` 不再按目录盲扫，而是按 `ImageSets/Segmentation/*.txt` 建索引
  - `VOC20` 的 `reduce_zero_label=True` 协议已经进入 `class_presence` 读取层
  - `run_feature_pipeline.py` 已支持 `--dataset-spec`
- 当前测试结果：
  - `pytest -q tests/test_dataset_feature_specs.py tests/test_voc20_subset_sampling.py tests/test_slice_remix_class_coverage.py tests/test_run_feature_pipeline_cli.py`
  - 结果：`10 passed`
- 当前解释：
  - 这一步还没有把 `VOC20` 全部训练/干预链跑起来
  - 但已经把“切 benchmark 时特征提取怎么迁移”的第一道工程 gate 通过了

## 2026-04-17 根目录 `feature` 过渡包已建立

- 当前已在 workspace 根目录建立：
  - `/home/yuhe/slicetune/feature`
- 当前目标：
  - 先把 feature extraction 相关的轻依赖公共契约层从 `clip_dinoiser` 中抽出一个 root-level 共享入口
  - 不在第一步就物理搬空 `feature_utils`
- 当前第一批暴露对象：
  - `feature.features.dataset_specs`
- 当前状态说明：
  - 这是 bridge package
  - 当前 source-of-truth 仍在 `clip_dinoiser.feature_utils.data_feature`
  - 但 root-level 公共导入入口已经存在并可用
- 当前验证：
  - `sys.path.insert(0, '/home/yuhe/slicetune')`
  - `from feature.features import get_dataset_feature_spec`
  - import 成功，并能读取 `voc20` spec

## 2026-04-17 `dataset_specs` 已提升为 root-level source-of-truth

- 当前 `feature` 已不只是 bridge package
- 第一批真正独立迁出的公共实现已经落地：
  - `/home/yuhe/slicetune/feature/features/dataset_specs.py`
- 当前 `clip_dinoiser` 内部对应模块已改成兼容桥接：
  - [dataset_specs.py](/home/yuhe/slicetune/clip_dinoiser/feature_utils/data_feature/dataset_specs.py:1)
- 当前回归验证：
  - `pytest -q tests/test_dataset_feature_specs.py tests/test_voc20_subset_sampling.py tests/test_slice_remix_class_coverage.py tests/test_run_feature_pipeline_cli.py tests/test_root_feature_bridge.py`
  - 结果：`11 passed`
- 当前解释：
  - 这标志着 feature extraction subsystem 的 workspace-level 迁移已经迈过第一道实质性门槛
  - 后续迁移可沿同样模式逐个模块推进，而不必一次性整体搬家

## 2026-04-17 workspace-level 公共包已统一更名为 `feature`

- 当前调整：
  - 将 `~/slicetune/slicetune_features` 重命名为 `~/slicetune/feature`
  - root import 统一为：
    - `from feature.features import ...`
  - repo-level bridge 与桥接测试已同步改名
- 当前目的：
  - 消除 `slicetune_features` / `feature` 两套命名并存带来的混淆
  - 为后续继续迁移 `bundle / postprocess / pipeline` 保持更短、更通用的公共包名

## 2026-04-17 官方 TensorFlow DeepLab 已完成最小可信复现

- 当前事实：
  - `~/slicetune/deeplab` 中的官方 DeepLab 代码已完成环境修复
  - `model_test.py` 已通过
  - `VOC2012` 官方 tar 下载与 `TFRecord` 转换已完成
  - 最小评测结果：
    - `eval/miou_1.0_overall = 0.821973264`
- 当前解释：
  - 这证明官方 DeepLab 链路在当前服务器上是可用、可信的
  - 当前在用户明确决策后，这条线不再只作为 reference
  - 后续 supervised segmentation 主实验将优先切到官方 DeepLab 框架
- 当前建议：
  - 接下来不再优先扩 `clip_dinoiser` 内的自实现 DeepLab runner
  - 优先研究如何把 feature-aware subset / intervention 接到官方 benchmark-specific 数据准备层
  - 主训练与评测继续使用官方 `train.py / eval.py`

## 2026-04-18 官方 DeepLab `VOC` lane 的 Phase 1 协议已明确

- 当前协议：
  - 训练池：`train_aug`
  - 固定评测集：`val`
  - 第一阶段不直接全量 `train_aug` 重训
  - 而是优先做固定预算子集对照：
    - `anchor`
    - `feature_high`
    - `feature_low`
    - `matched_random`
- 当前理由：
  - `train_aug` 更接近官方强 recipe
  - 固定 `val` 有利于本地可重复评测
  - 固定预算子集更适合先验证 feature signal，而不是直接追最终最高分

## 2026-04-18 官方 DeepLab `VOC train_aug` 资产已实际落盘

- 当前已完成：
  - 下载 `train_aug.txt` 到：
    - `/home/yuhe/slicetune/deeplab/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/ImageSets/Segmentation/train_aug.txt`
  - 下载并解压 `SegmentationClassAug.zip` 到：
    - `/home/yuhe/slicetune/deeplab/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/SegmentationClassAug`
- 当前清理：
  - 已移除 zip 自带的 `__MACOSX`
  - 已移除 `SegmentationClassAug/` 内的 `._*` 垃圾文件
- 当前核验结果：
  - `train_aug.txt` 行数：`10582`
  - `train_aug` ids 缺失 mask：`0`
  - `SegmentationClassAug` 内 `png` 文件总数：`12031`
- 当前解释：
  - 这里的 `12031` 表示增强标注目录中总 mask 数量，不等于最终参与 `train_aug` 的样本数
  - 真正训练池仍由 `train_aug.txt` 决定；当前只确认所有 `10582` 个 `train_aug` id 都已被增强标注覆盖
- 当前剩余 gate：
  1. `SegmentationClassAug -> SegmentationClassAugRaw`
  2. `build_voc2012_data.py` 生成 `train_aug-*.tfrecord`
  3. 在完整 `train_aug` 上实现并提取 `rare_class_coverage`
  4. 固定 `anchor_2000` 并生成 `high/low/matched_random` split

## 2026-04-18 官方 `train_aug` 转换已进入执行

- 当前已执行的官方转换命令逻辑：
  1. `remove_gt_colormap.py`
     - `SegmentationClassAug -> SegmentationClassAugRaw`
  2. `build_voc2012_data.py`
     - `semantic_segmentation_folder=SegmentationClassAugRaw`
     - `list_folder=ImageSets/Segmentation`
- 当前状态：
  - `SegmentationClassAugRaw` 已生成完成，当前 `png` 文件数为 `12031`
  - `build_voc2012_data.py` 正在运行，会按官方逻辑把 `ImageSets/Segmentation/*.txt` 全部转换成对应 TFRecord
- 当前解释：
  - 因为 `build_voc2012_data.py` 会 `Glob(list_folder/*.txt)`，所以这一轮不只会生成 `train_aug-*.tfrecord`，也会一并重写：
    - `train-*.tfrecord`
    - `val-*.tfrecord`
    - `trainval-*.tfrecord`
  - 这没有问题，因为当前 `SegmentationClassAug` 已覆盖：
    - `train`
    - `val`
    - `trainval`
    - `train_aug`

## 2026-04-18 官方 `VOC train_aug` TFRecord 已全部就绪

- 当前用户已在终端确认：
  - `train_aug-*.tfrecord` 数量：`4`
  - 四套 split shard 数：
    - `train = 4`
    - `val = 4`
    - `trainval = 4`
    - `train_aug = 4`
- 当前解释：
  - 官方 DeepLab `VOC train_aug` 数据准备链路已完全通过
  - 当前后续不再需要继续调 VOC augmentation 数据
  - 当前下一阶段正式切到：
    - `anchor_2000`
    - 以及后续 feature-aware `high/low/random` split 构造

## 2026-04-18 官方 DeepLab 已支持 custom split TFRecord

- 当前触发点：
  - 用户首次用：
    - `--train_split=anchor_2000_seed0`
  - 官方 `data_generator.py` 报错：
    - `ValueError: data split name anchor_2000_seed0 not recognized`
- 当前原因定位：
  - 官方 `pascal_voc_seg` 只白名单：
    - `train`
    - `train_aug`
    - `trainval`
    - `val`
  - 但我们已经正确生成了：
    - `anchor_2000_seed0-00000-of-00004.tfrecord`
    - ...
- 当前修复：
  - 在官方 `data_generator.py` 中加入最小兼容：
    - 若 `<split>-*.tfrecord` 已存在，则允许该 split 继续训练
- 当前解释：
  - 这不是改训练算法
  - 只是把数据层的 split 白名单扩展为：
    - 内置官方 split
    - 或已有 TFRecord 支持的自定义 split

## 2026-04-18 官方 DeepLab GPU 运行时阻塞定位

- 当前触发点：
  - 用户开始用官方 DeepLab 在 `anchor_2000_seed0` 上训练后，观察到：
    - 每个 step 大约 `7s+`
  - 这明显不符合 4090 上 `2000` 图 VOC 子集训练的预期速度
- 当前实机诊断：
  - `tensorflow-gpu==1.15.5`
  - `protobuf==3.20.3`
  - `tf_slim==1.1.0`
  - `tf.test.is_gpu_available()` 返回 `False`
  - `device_lib.list_local_devices()` 仅显示：
    - `/device:CPU:0`
    - `/device:XLA_CPU:0`
    - `/device:XLA_GPU:0..3`
  - TensorFlow 日志明确报缺：
    - `libcudart.so.10.0`
    - `libcublas.so.10.0`
    - `libcufft.so.10.0`
    - `libcurand.so.10.0`
    - `libcusolver.so.10.0`
    - `libcusparse.so.10.0`
    - `libcudnn.so.7`
  - 并输出：
    - `Cannot dlopen some GPU libraries`
    - `Skipping registering GPU devices...`
- 当前解释：
  - 当前服务器驱动本身没问题，TensorFlow 也能看到 `libcuda.so.1`
  - 但 `TF1.15` 预编译 wheel 期待的是 `CUDA 10.0 + cuDNN 7` 时期的用户态运行时库
  - 当前环境只暴露了现代驱动和 `/usr/local/cuda/lib64`，因此 GPU 最终没有被 TensorFlow 正式注册
  - 这解释了为什么：
    - `local_test` 能跑
    - 但正式训练速度仍非常慢
- 当前结论：
  - 当前官方 DeepLab 的真正 blocker 已经从：
    - 数据准备 / custom split
  - 转移为：
    - `TF1 GPU runtime` 不完整
- 当前下一步：
  1. 优先尝试为 `deeplab` conda 环境补齐：
     - `cudatoolkit=10.0.130`
     - `cudnn=7.6.5`
  2. 重新跑 GPU 可用性检查
  3. 若仍不能稳定注册 GPU，则不应继续在该状态下大规模跑 `anchor/high/low/random`

## 2026-04-18 官方 DeepLab GPU 运行时修复完成

- 当前结果：
  - 用户已完成：
    - `conda install cudatoolkit=10.0.130 cudnn=7.6.5`
    - `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"`
  - 重新检查后：
    - `tf.test.is_gpu_available() = True`
    - `device_lib.list_local_devices()` 已出现：
      - `/device:GPU:0`
      - `/device:GPU:1`
      - `/device:GPU:2`
      - `/device:GPU:3`
  - TensorFlow 日志已成功加载：
    - `libcudart.so.10.0`
    - `libcublas.so.10.0`
    - `libcufft.so.10.0`
    - `libcurand.so.10.0`
    - `libcusolver.so.10.0`
    - `libcusparse.so.10.0`
    - `libcudnn.so.7`
- 当前解释：
  - 这说明官方 TensorFlow DeepLab 当前已经真正具备 GPU 训练条件
  - 之前 `7s+/step` 的慢速问题，主要来源于旧 CUDA 运行时缺失导致的 CPU 回退
  - 当前后续应重新跑：
    - `anchor_2000_seed0`
    并用真实单卡 GPU 速度作为后续 feature-aware 实验预算基准
- 当前新的最小操作规范：
  - 每次训练或评测前先执行：
    - `conda activate deeplab`
    - `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"`

## 2026-04-18 官方 DeepLab GPU 训练仍不稳定

- 当前新增事实：
  - 在 GPU runtime 修复后：
    - 自定义 `anchor_2000_seed0`
    - 官方 `trainval`
    两条训练 smoke 都没有顺利进入稳定 step/loss 前进
  - 已观察到的异常模式包括：
    - 长时间停在 `global_step/sec: 0`
    - 首步即 `Loss is inf or nan`
- 当前已排除：
  - custom split 未识别
  - `anchor_2000_seed0` 标签值异常
  - VOC segmentation mask 脏值
  - `cudatoolkit/cudnn` 缺失
- 当前更合理的解释：
  - 现在的问题已经不主要像是数据协议 bug
  - 更像是：
    - `TF1.15 + CUDA10.0/cuDNN7 + Ada(4090)` 在训练/反向传播阶段存在兼容性或数值稳定性问题
  - 这一点尤其因为：
    - 连官方内置 `trainval` split 的纯官方 smoke 也表现异常
- 当前下一步：
  1. 做一次 CPU 对照 smoke
  2. 若 CPU 正常、GPU 异常，则正式将 root cause 收敛到 GPU 训练栈兼容性

## 2026-04-18 官方 DeepLab 训练 NaN 的直接根因已定位

- 当前新增调试动作：
  - 在官方 `deeplab/train.py` 中加入：
    - `--debug_input_tensors`
    - `--debug_loss_tensors`
  - 在 `train_utils.add_softmax_cross_entropy_loss_for_each_scale(...)` 中加入：
    - `image/label` 统计输出
    - `logit_min/logit_max`
    - `pixel_loss_min/pixel_loss_max`
    - `num_present/total_loss/loss`
- 当前关键对照：
  - **CPU-only**
    - `CUDA_VISIBLE_DEVICES=''`
    - `trainval`
    - `batch_size=1`
    - `fine_tune_batch_norm=false`
    - `base_learning_rate=1e-5`
    - 结果：
      - 输入/标签统计正常
      - 第 1 步 `logits` 有限
      - `pixel_loss` 有限
      - `loss` 有限
      - 1-step 正常结束
  - **GPU**
    - 同样的 `trainval + batch_size=1 + fine_tune_batch_norm=false + base_learning_rate=1e-5`
    - 第 1 步可以得到有限 loss
    - 随后 TensorFlow 报：
      - `Detected cudnn out-of-bounds write in convolution buffer!`
      - `This is likely a cudnn bug`
    - 下一步 debug 输出中：
      - `logit_min = 3.40282347e+38`
      - `logit_max = -3.40282347e+38`
      - `total_loss = nan`
      - `loss = nan`
- 当前收敛结论：
  - NaN 的真正来源已从“数据或标签协议错误”收敛为：
    - **老 TF1/cuDNN7 在 Ada(4090) 上选择的卷积算法发生 buffer 越界写，污染 GPU state，随后导致 logits/loss NaN**
  - 这解释了为什么：
    - GPU runtime 已修好
    - 数据检查也正常
    - 但训练仍不稳定
- 当前研究含义：
  - 官方 DeepLab 代码库可以保留为：
    - 参考实现
    - eval / recipe sanity lane
  - 但在当前硬件上，不应继续把它当成 SliceTune 的高频主训练底座

## 2026-04-18 modern PyTorch supervised probe 重新接回主线

- 当前新增工程动作：
  - 将 `research_harness/supervised_probe.py` 从 COCO-Stuff 专用逻辑扩展为 dataset-aware runner
  - 当前已支持：
    - `coco_stuff`
    - `voc20`
    - `cityscapes`
  - 当前已支持两种训练入口：
    - full-train baseline（不传 manifest）
    - manifest-defined subset 训练
  - `run_supervised_probe_experiment.py` 已新增：
    - `--dataset`
    - `--data-root`
    - optional `--subset-manifest`
- 当前验证：
  - 新增 `tests/test_supervised_probe_dataset_cfg.py`
  - 在 `clipdino2` 环境下通过：
    - `python -m unittest tests.test_supervised_probe_dataset_cfg`
- 当前研究含义：
  - supervised segmentation lane 的下一步不再是继续调官方 TF1 DeepLab
  - 而是先在 modern PyTorch / mmseg 上跑：
    - `voc20` full baseline
    - `cityscapes` full baseline
  - baseline 稳定后，再把 feature-aware subset 干预接回主线

## 2026-04-18 VOC baseline 首轮结果暴露出 recipe mismatch

- 当前观察：
  - 首轮 `voc20` full baseline 输出：
    - `subset_size=1464`
    - `max_iters=2000`
    - `samples_per_gpu=2`
    - `mIoU=4.68`
  - 这说明首轮实际使用的是：
    - `train.txt`
    - 而不是 `train_aug`
- 当前解释：
  - 该结果暂不能直接与：
    - 官方 TensorFlow DeepLab 的 `80%+`
    - 或 mmseg 的标准 VOC12aug 结果
    直接比较
  - 主要 mismatch 至少包括：
    - 训练池：`1464` vs `10582`
    - schedule：`2000 iter` vs `20k/40k`
    - 有效 batch：`2` vs 多卡 recipe
    - benchmark 口径：`voc20` vs 标准 `voc`（21 类）并不完全相同
- 当前工程修正：
  - `voc20` 现已改为：
    - 若检测到 `train_aug.txt + SegmentationClassAug`，则优先使用增强训练池
  - 同时新增标准：
    - `voc` dataset key（21 类、含背景）
- 当前下一步：
  - 重新跑：
    - `voc20` with `train_aug`
    - `voc` full baseline
  - 在 recipe 更接近标准设定后再判断是否仍存在实现级 bug

## 2026-04-18 modern PyTorch `voc train_aug` baseline 已站住

- 当前结果：
  - `dataset=voc`
  - `subset_size=10582`
  - `max_iters=20000`
  - `samples_per_gpu=8`
  - `mIoU=73.51`
  - `mAcc=86.48`
  - `aAcc=93.64`
  - `total_seconds=8408.018`（约 `2.34h`）
- 当前解释：
  - 这说明：
    - modern PyTorch / mmseg DeepLabV3+ 训练链已经足够稳定
    - VOC train_aug lane 已可作为 supervised feature-sensitivity 诊断底座
  - 当前主要问题已不再是“有没有大 bug”，而是：
    - 是否需要进一步逼近公开 recipe
    - 以及 feature experiment 的预算应如何从 `20k` 下调到更高吞吐的 screening budget
- 当前执行建议：
  - 将 `20k` 保留为：
    - full baseline
    - confirmatory run
  - 将后续 feature intervention 首轮预算降为：
    - `5k`
    - 或 `10k`
  - 优先回答：
    - 哪个最小预算下仍能稳定观察到 `real > control`

## 2026-04-18 modern PyTorch supervised probe 已补多卡分布式路径

- 当前新增工程动作：
  - `research_harness/supervised_probe.py` 已从单卡路径扩展为同时支持：
    - 单卡 `launcher=none`
    - 多卡 distributed `launcher=pytorch/slurm/mpi`
  - 当前训练阶段已支持：
    - `train_segmentor(..., distributed=True)`
  - 当前评测阶段已支持：
    - `multi_gpu_test(...)`
    - 仅 rank0 汇总并写出最终结果
  - `run_supervised_probe_experiment.py` 已新增：
    - `--launcher`
    - `--dist-backend`
    - `--gpu-collect`
    - `--local-rank/--local_rank`
- 当前验证：
  - `tests/test_supervised_probe_dataset_cfg.py` 已新增 distributed CLI 覆盖
  - `python -m unittest tests.test_supervised_probe_dataset_cfg` 通过
- 当前研究含义：
  - 当前 supervised segmentation 主线已具备：
    - 单卡可用 baseline
    - 多卡可用执行入口
  - 当前下一步不再是“能不能多卡”，而是：
    - 先选定 screening budget
    - 再在 `anchor/high/low/random` 矩阵上利用多卡提高吞吐

## 2026-04-18 首轮 DDP smoke 的现实 blocker 已定位

- 当前观察：
  - 首轮 `python -m torch.distributed.run --nproc_per_node=4 ...` 已使用正确的 `clipdino2` Python
  - 但训练在 rank `3` 第一层卷积处报：
    - `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`
- 当前排查：
  - 机器上已有外部长任务占用全部 `4` 张 GPU
  - 进程信息：
    - `pid=2772903`
    - 用户：`wangjy`
    - 命令：`python train_reasoning.py`
  - 其显存占用大致为：
    - GPU0: `24.3GB`
    - GPU1: `22.6GB`
    - GPU2: `22.5GB`
    - GPU3: `47.2GB`
  - 因此首轮 `4` 卡 DDP 失败的更直接原因不是 runner 逻辑，而是：
    - 第 `3` 号卡几乎无剩余显存，worker 无法正常初始化 cuDNN
- 当前执行建议：
  - 若继续做 DDP smoke，应先改为：
    - `2` 卡对齐实验
    - 或 `3` 卡低 batch smoke
  - 待外部进程释放 GPU 后，再回到正式 `4` 卡对齐 benchmark

## 2026-04-18 distributed runner 已改为保留 SyncBN

- 当前新增工程动作：
  - `build_supervised_probe_cfg(...)` 已新增：
    - `preserve_syncbn`
  - 当前行为变为：
    - 单卡路径：仍将 `SyncBN -> BN`
    - distributed 路径：保留原始 `SyncBN`
- 当前验证：
  - `tests/test_supervised_probe_dataset_cfg.py` 已新增对应测试
  - `python -m unittest tests.test_supervised_probe_dataset_cfg` 通过
  - 直接检查 config：
    - 单卡：`backbone.norm_cfg.type == 'BN'`
    - distributed：`backbone.norm_cfg.type == 'SyncBN'`
- 当前研究含义：
  - 首轮 `2` 卡 `200 iter` smoke 的低精度现在更应解释为：
    - 训练步数极短
    - 外部 GPU 资源拥塞
    - 以及修正前 distributed recipe 仍被降级为普通 `BN`
  - 后续若继续判断 DDP 精度行为，应以“保留 `SyncBN` 后的新 smoke”作为准绳

## 2026-04-18 2卡 `SyncBN` smoke 已完成

- 当前结果：
  - `dataset=voc`
  - `max_iters=200`
  - `samples_per_gpu=4`
  - 总 batch=`8`
  - `mIoU=9.94`
  - `mAcc=14.08`
  - `aAcc=78.87`
  - `train_seconds=63.687`
  - `eval_seconds=22.949`
  - `total_seconds=89.307`
- 当前解释：
  - 该结果说明：
    - distributed path + `SyncBN` 已能稳定训练、评测并写出结果
    - `rank0` 汇总路径正常
  - 但与此前 `2` 卡 `200 iter` 普通 `BN` smoke 相比，指标形态几乎一致
  - 因而当前更合理的解释是：
    - `200 iter` 主要验证 DDP 工程可行性
    - 还不足以评估 distributed recipe 的精度收益
- 当前下一步：
  - 若要正式比较：
    - 单卡 vs `2` 卡
    - 推荐统一跑 `1000 iter`
  - 若要进入 feature screening：
    - `2` 卡 DDP 已可作为执行底座
    - 但 screening budget 建议不低于 `1000 iter`

## 2026-04-19 full `train_aug` baseline 之后的实验主线已收敛

- 当前新增判断：
  - 既然 `voc + train_aug` full baseline 和 `2` 卡 `1000 iter` DDP 都已证明：
    - learner 能正常学起来
    - 现代 PyTorch 训练/评测链稳定
    - DDP 具备可用吞吐
  - 下一步不应继续只是堆更多 full-pool baseline
  - 而应回到当前 Phase 1 核心问题：
    - `train_aug` 中哪些 feature 真会带来可重复的训练响应
- 当前主线实验顺序：
  1. 在完整 `train_aug` 候选池上提取目标 feature
  2. 基于全池 feature 构造受控 subset：
     - `anchor`
     - `high`
     - `low`
     - `matched_random`
  3. 先用 `2` 卡 DDP + `1000 iter` 做 pilot screening
  4. 若某 feature 轴出现明确响应，再升到 `5000 iter` 做确认
- 当前默认首批 feature 轴：
  - `small_object_ratio`
  - `rare_class_coverage`

## 2026-04-19 单卡 vs 2卡 `1000 iter` 对照已完成

- 当前结果：
  - 单卡：
    - `mIoU=53.13`
    - `mAcc=70.20`
    - `aAcc=88.32`
    - `train_seconds=427.59`
    - `total_seconds=462.597`
  - `2` 卡 DDP + `SyncBN`：
    - `mIoU=52.10`
    - `mAcc=67.58`
    - `aAcc=88.65`
    - `train_seconds=283.651`
    - `total_seconds=308.855`
- 当前解释：
  - 在 `1000 iter` 这个 pilot 预算下：
    - `2` 卡 DDP 与单卡已基本精度对齐
    - 同时带来约 `1.5x` wall-clock 提升
  - 这足以说明：
    - DDP 不只是“工程上能跑”
    - 也已经足够作为 feature screening 的默认执行底座
- 当前结论：
  - 当前无需再在单卡 vs 2卡 上继续反复纠缠
  - 下一步应真正回到主问题：
    - 在完整 `train_aug` 上提目标 feature
    - 物化 `anchor/high/low/matched_random`
    - 用 `2` 卡 `1000 iter` 做首轮响应审计

## 2026-04-19 VOC `train_aug` feature-prep 代码已准备完成

- 当前新增代码：
  - 包层服务：
    - `slice_remix/voc_feature_subsets.py`
  - 薄 CLI：
    - `tools/prepare_voc_train_aug_feature_experiment.py`
- 当前新增能力：
  - 读取完整 `train_aug` 池
  - 计算：
    - `small_object_ratio`
    - `rare_class_coverage`
  - 写出：
    - `feature_table.jsonl`
    - `summary.json`
    - `manifest_index.json`
    - `manifests/*.json`
  - 物化：
    - `anchor`
    - `high`
    - `low`
    - `matched_random`
- 当前验证：
  - `python -m unittest tests.test_voc_feature_subsets tests.test_supervised_probe_dataset_cfg`
    - `OK`
  - `python tools/prepare_voc_train_aug_feature_experiment.py --help`
    - 正常
- 当前含义：
  - 现在已经可以直接运行 full `train_aug` feature-prep
  - 下一步不再是“补准备代码”，而是实际生成第一版 VOC feature screening artifacts

## 2026-04-19 `small_object_ratio` pilot 已完成

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
  - 该结果已经证明：
    - 当前 learner 对 subset 组成变化敏感
    - 响应幅度远高于当前噪声水平
  - 但当前不支持把 `small_object_ratio` promote 为 clean monotonic feature axis：
    - `high` 和 `low` 都明显低于 `anchor`
    - `matched_random` 反而略高于 `anchor`
  - 当前最合理的解释是：
    - 现有 `small_object_ratio` intervention 触发了强 response
    - 但该 response 目前更像 extreme subset construction / off-target drift，而不是单纯 feature 本身的单调效应
- 当前补充诊断：
  - 每图前景类数均值：
    - `anchor=1.470`
    - `high=1.544`
    - `low=1.163`
    - `matched_random=1.485`
  - `low` 显著降低了每图前景类数
  - `high` 则提高了 `person` presence，并带来明显类存在模式偏移
- 当前结论：
  - 这轮结果应解释为：
    - `response signal exists`
    - `axis validity not yet established`
  - 下一步优先：
    - 跑 `rare_class_coverage` pilot
    - 若要继续用 `small_object_ratio`，先收紧 `matched_random` 与同轴互斥/匹配约束

## 2026-04-19 研究目标已明确转向“非单调、多特征耦合、偏好 insight”

- 当前用户澄清：
  - 项目并不以“找到单调 feature 轴”为主要目标
  - 真实目标更接近：
    - 多特征耦合下的复杂分布优化
    - 识别过高/过低都不理想的非单调区间
    - 通过人机交互，为用户提供当前训练任务更适合偏向哪些数据特征的 insight
- 当前对既有结果的重新解释：
  - `small_object_ratio` pilot 中：
    - `high/low` 都差
    - `matched_random` 更好
  - 这不应直接读成“feature 无效”
  - 它反而更像在支持：
    - 单轴极端推进会破坏整体混合分布
    - 更优状态可能来自若干 feature 的中间组合
- 当前执行含义：
  - 后续 Phase 1 分析不再只寻找单轴单调关系
  - 应优先积累：
    - 非单调 response 证据
    - 多轴耦合证据
    - 用户可用的 feature distribution steering insight

## 2026-04-19 VOC pilot control 已收紧并新增结果汇总工具

- 当前新增代码：
  - `slice_remix/voc_feature_subsets.py`
    - `matched_random` 现在与同轴 `anchor/high/low` 互斥
    - `summary.json` 会记录 overlap 统计
  - `tools/summarize_supervised_probe_results.py`
    - 可对一组 `result.json` 直接输出相对某个 reference 的 summary/delta
- 当前验证：
  - `python -m unittest tests.test_voc_feature_subsets tests.test_supervised_probe_dataset_cfg`
    - `OK`
  - `python tools/summarize_supervised_probe_results.py --help`
    - 正常
- 当前执行含义：
  - 若要继续 `rare_class_coverage` pilot
  - 或重跑更干净的 `small_object_ratio` pilot
  - 应先重新生成一版 VOC feature-prep artifacts

## 2026-04-19 `rare_class_coverage` strict pilot 已完成

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
  - 该结果与 `small_object_ratio` pilot 一样，不支持“单轴单调更好”
  - 但它更清楚地支持：
    - 当前任务存在 balanced mixture 优于极端偏置的结构
    - `anchor` 显著优于 `high/low/matched_random`
- 当前补充诊断：
  - `high` subset：
    - `mean_fg_classes=2.184`
    - `mean_small_object_ratio=0.355`
  - `low` subset：
    - `mean_fg_classes=1.034`
    - `mean_small_object_ratio=0.088`
  - 当前说明：
    - `rare_class_coverage` 的极端推进仍强烈耦合了别的复杂度轴
    - 当前 response 更像混合分布被破坏，而不是单轴本身的可分离效应
- 当前结论：
  - 这轮结果应被解释为：
    - `balanced anchor > extreme subsets`
    - `non-monotonic mixture hypothesis supported`
  - 下一步优先：
    - 不再只做极端 `high/low`
    - 转向中等幅度 perturbation
    - 转向两轴局部联合实验

## 2026-04-19 最近实验事实汇总文档已补充

- 当前新增文档：
  - `.slicetune/logs/2026-04-19_recent_experiment_fact_sheet.md`
- 当前覆盖内容：
  - 最近 VOC supervised probe 实验及结果
  - 训练算法、训练协议、评测协议、数据集和任务场景
  - 当前与历史自定义特征指标及语义

## 2026-04-21 下一阶段 feature validation 方案已收敛

- 当前收敛后的执行框架：
  1. coarse screening
  2. strict validation
  3. local interaction / shape experiment
- 当前不再采用：
  - 单个特征一路线性深挖到底
- 当前建议候选池规模：
  - `10–12` 个候选特征
- 当前建议主系统冻结规模：
  - 约 `6` 个特征
- 当前工程优先事项：
  - 把 VOC feature-prep 从“当前两条轴专用实现”推广为：
    - generic feature registry
    - generic feature table extraction
    - generic subset materialization

## 2026-04-21 VOC feature-prep 已完成第一轮包层重构

- 当前完成的结构调整：
  - `slice_remix/voc_feature_prep/contracts.py`
  - `slice_remix/voc_feature_prep/dataset.py`
  - `slice_remix/voc_feature_prep/scoring.py`
  - `slice_remix/voc_feature_prep/service.py`
  - `slice_remix/voc_feature_prep/__init__.py`
- 当前旧入口状态：
  - `slice_remix/voc_feature_subsets.py`
    - 已改为 backward-compatible bridge
- 当前新增能力：
  - `tools/prepare_voc_train_aug_feature_experiment.py`
    - 新增 `--feature-axis`
    - 可只物化指定 axis
- 当前验证：
  - `python -m unittest tests.test_voc_feature_subsets tests.test_supervised_probe_dataset_cfg`
    - `OK`
  - `python tools/prepare_voc_train_aug_feature_experiment.py --help`
    - 正常

## 2026-04-21 VOC feature registry 已接入新一批 mask-native 轴

- 当前默认 screening 轴保持不变：
  - `small_object_ratio`
  - `rare_class_coverage`
- 当前新接入可选轴：
  - `foreground_class_count`
  - `pixel_class_entropy`
  - `foreground_area_ratio`
  - `foreground_component_count`
  - `component_fragmentation`
- 当前设计含义：
  - 先把更贴近 segmentation 标签结构的 mask-native 轴接进统一 registry
  - 之后再接：
    - `laplacian`
    - `bga`
    - `noise_pca`
    等 image-quality 族
