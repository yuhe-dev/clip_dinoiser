# 研究执行手册

本文件是 `.slicetune/` 的主规则文件，融合了：

- pipeline 契约
- benchmark 与证据契约
- 工程规范与结构边界
- 自动科研与 agent 运行规则

目标是尽量用一个高频文件回答四个问题：

1. 当前 pipeline 每一段负责什么
2. 什么结果算可比、什么证据足以晋升
3. 代码应该如何继续收敛
4. agent system 实际如何被约束和运行

---

## 1. Pipeline 执行契约

### Stage 1：Feature Extraction

目标：

- 把样本转换为可分析、可比较、可聚合的 secondary feature 表达

当前关键问题：

- 当前 feature 是否真的与训练结果相关
- 当前 feature 是否只是“可解释”，但不可干预

### Stage 2：Feature Embedding / Clustering / Slice Construction

目标：

- 把样本级 feature 压缩成可操作的 slice 表达

当前关键问题：

- 哪种 slice 定义最有干预杠杆
- K 应按内部相似性选，还是按可干预性选

### Stage 3：Human Preference-Aware Candidate Generation

目标：

- 在固定预算下生成可执行、可解释、可排序的 remix 候选

当前关键问题：

- objective 应该是什么
- “逼近 full50k feature 分布”是否是可靠 heuristic

### Stage 4：Materialization / Practical Validation

目标：

- 把 target mixture 变成真实 subset，并量化 realized-vs-target drift

当前关键问题：

- 偏差来自目标本身还是 materialization 噪声
- 改变的是否真是目标特征，而不是别的混杂变量

### Stage 5：Surrogate / Refinement / Iteration

目标：

- 在 response dataset 足够可靠时，对候选做预测、排序与下一轮建议

当前关键问题：

- 当前 response dataset 的动态范围是否足够支撑 surrogate

### 总规则

- 如果某一阶段尚未形成足够证据，下游阶段默认只能做影子分析、最小接口预留或最小验证。
- 不允许用下游复杂化去掩盖上游 response signal 不足。

## 1.1 Research-Native Task State Machine

后续 task-level conductor 的状态机采用研究版状态序列：

1. `hypothesis`
2. `design`
3. `audit`
4. `execution`
5. `verification`
6. `judgment`
7. `acceptance`

各 loop 的 `task_plan.json` 需显式记录 `research_state / next_state`，task board 会验证其合法性。

---

## 2. Benchmark 与证据规则

### 当前主 benchmark

- 训练池：`COCO-Stuff` 相关训练子集
- 主验证：`COCO-Stuff val`
- 当前主锚点 learner：`CLIP-DINOiser` 系列训练协议

### 当前预算层级

- `Tier A`：低成本快速筛选
- `Tier B`：中成本重复确认
- `Tier C`：高成本正式比较

### 指标层级

一级指标：

- global `mIoU`

二级指标：

- per-class IoU
- focus class / long-tail class 指标
- failure slice 指标
- realized-vs-target fidelity

三级诊断指标：

- 训练噪声
- 实例化噪声
- 结果方差
- candidate ranking regret

### 可比较结果的最低要求

一个结果要进入正式决策，默认至少满足：

1. benchmark 版本一致
2. 训练协议版本一致，或变化被明确记录
3. seed policy 明确
4. artifact 完整
5. 可追溯到 experiment card

### 证据等级

- `E0`：直觉 / 文献启发
- `E1`：单次可运行结果
- `E2`：低成本重复结果
- `E3`：中高成本稳定结果
- `E4`：跨设置仍稳定的结果

### Promote 规则

一个方向默认只有达到至少 `E3` 才允许正式 promote。

Promote 至少要求：

1. 与 champion 比较口径一致
2. 改善超过 noise floor
3. 无明显协议污染
4. artifact 完整
5. Judge 明确 promote

### Park / Kill 规则

适合 `park` 的情况：

- 有信号但不稳定
- 有趣但不属于当前 phase
- 接入成本过高且暂时不值得主线投入

适合 `kill` 的情况：

- 连续多轮未超过 noise floor
- 理论目标被证伪
- 相比简单 baseline 没有稳定优势
- 成本高但收益长期过低

### 当前项目的特殊例外

如果 global `mIoU` 动态范围很窄，则以下情况允许进入更高一级审查：

- focus classes 明显改善
- failure slices 指标明显改善
- realized-vs-target fidelity 明显改善

但前提是：

- 不出现明显全局退化
- 改善具有重复性

---

## 3. 工程规范与结构边界

### 工程总原则

- 先分层，再扩功能
- 高内聚、低耦合
- I/O 与计算分离
- 业务逻辑进入包层，CLI 保持薄包装
- 新功能优先进已有包，不再默认新增同类顶层小脚本

### 推荐分层

新增功能前先判断它属于哪一层：

- domain logic
- application / service layer
- CLI layer
- analysis / report layer
- state / memory layer

### 目标代码结构

后续希望逐步收敛到：

```text
clip_dinoiser/
  AGENTS.md
  .slicetune/
  configs/
  docs/
  tests/
  tools/
    experimental/

  feature_utils/
  slice_discovery/
  slice_remix/
  research_harness/

  run_*.py
  analyze_*.py
```

### 包级边界

`feature_utils` 负责：

- feature 定义、提取、编码、bundle、pipeline

`feature_utils` 不负责：

- slice clustering
- candidate generation
- surrogate

`slice_discovery` 负责：

- assembler
- projector
- clustering / finder
- K selection
- stability / leverage 诊断

`slice_discovery` 不负责：

- objective
- materialization
- surrogate

`slice_remix` 负责：

- baseline
- candidate / objective / preference
- materialization
- response dataset
- surrogate
- analysis / reports

`slice_remix` 的关键边界：

- objective 与 search 分离
- target 与 realization 分离
- surrogate 与 truth 分离

`research_harness` 负责：

- contracts
- queue
- judge
- dispatch / summarize
- runtime helpers
- runtime profile registry
- worker preflight
- lease / heartbeat / resume 约束
- completion sentinel 与 session error 产物

`research_harness` 不负责：

- 具体算法实现

### 脚本治理

顶层脚本允许保留：

- 稳定 CLI 入口
- 分析入口
- legacy 兼容入口

不应继续扩张：

- 一次性分析脚本
- 只是参数不同的复制脚本
- 主线功能的临时变体

临时脚本应进入：

- `tools/experimental/`

并登记到 `board.md` 的工程待办。

### 测试与验证

验证层级：

- `V0`：静态检查
- `V1`：单元测试
- `V2`：集成 smoke test
- `V3`：实验级验证
- `V4`：高成本正式确认

最低要求：

- 新增模块：至少单测或 smoke test
- 新增 CLI：至少 `--help` 和最小 happy path
- 修改数据契约：必须补 schema / round-trip 测试

### 变更控制

变更类型：

- A 类：文档 / 状态更新
- B 类：工程重构
- C 类：算法改动
- D 类：评测协议改动

默认禁止：

- 同一轮同时改 benchmark 与算法并直接解释结果
- 同一轮大规模搬家 + 新算法 + 新协议混在一起
- 用工程重构结果包装成算法改进

---

## 4. 自动科研与 Agent 运行

### 运行模型

仅靠“每次给 agent 一个 prompt”是不够的。

完整运行分四层：

1. `Prompt / AGENTS.md`
2. `.slicetune/` 记忆与状态层
3. `research_harness/` 执行与判决层
4. `tmux / cron / systemd` 等后台运行层

### Runtime 约束

所有需要真实 worker 的 loop，默认必须满足：

1. worker runtime 通过 `runtime profile` 选择，不依赖裸 `python` 字符串或 shell PATH 猜测
2. controller 在执行前必须完成 preflight
3. preflight 至少检查：
   - python executable
   - worker script 存在
   - config 存在
   - 关键模块 import
   - CUDA 可见性（若任务要求 GPU）
4. preflight 未通过时，卡片进入 `blocked_preflight`，不得进入 `running`
5. 可恢复任务默认需要成功 sentinel，不能只凭输出文件存在判断已完成
6. worker 异常必须被 controller 收口到明确终态，不允许长期停在 `running`

### Scheduler 与 Daemon 约束

自动科研循环默认遵循：

1. scheduler 至少按 `priority + depends_on` 选择下一张卡
2. 若依赖未完成，下游卡不得进入 `running`
3. 每次正式执行都必须生成独立 `attempt`
4. attempt 至少要保留：
   - attempt manifest
   - card snapshot
   - session / preflight / judge / run manifest 的证据路径
5. daemon 默认只在三种情况下继续：
   - 成功完成且仍有 ready card
   - 当前无 ready card，但允许进入 idle wait
   - policy 允许继续跳过失败卡
6. daemon 默认必须在以下情况下停止：
   - `awaiting_human_review`
   - `phase_completion_candidate` 成功并触发 stop gate
   - fatal failure 且 policy 不允许继续

### Proposer 约束

1. proposer 默认只能读取：
   - `runtime_index`
   - `program.md`
   - `proposal_policy`
   - 已有 experiment cards
2. proposer 默认只能沿着当前 phase 已锁定的 follow-on rule 生成提案
3. proposer 不得自由发明新的主 research branch
4. proposal 分两类：
   - `safe_auto_card`：可自动 materialize 为卡片
   - `draft_only`：只生成 `planned` 卡片或 proposal record，不能直接进入运行
5. 重大 proposal 默认仍需 debate / human review，除非 proposal policy 明确允许自动落卡

### Retry 约束

1. 卡片可设置 `max_attempts`
2. 当 `attempt_count >= max_attempts` 时，controller 默认转为 `blocked_retry_limit`
3. 若要继续，必须显式提高 `max_attempts` 或重新定义卡片

当前已具备：

- 规范层
- 记忆层

后续要补：

- queue
- dispatch
- judge
- summary
- 后台 controller / worker

### 长时运行原则

如果目标是连续工作数小时，默认必须具备：

1. session checkpoint
2. 可恢复 handoff
3. 独立 reviewer / judge
4. run manifest
5. 独立 judge policy
6. 周期性 context compaction

不满足这些条件时，不应把系统当成真正的长时 autonomous research loop。

### Memory 协议

长时 session 默认只常驻加载：

1. `AGENTS.md`
2. `.slicetune/MEMORY.md`
3. `.slicetune/context/program.md`
4. `.slicetune/context/playbook.md`
5. `.slicetune/state/board.md`
6. `.slicetune/state/decision_log.md`

细节材料按需读取，避免把所有历史内容塞进单次上下文。

### 角色拓扑

`v1` 常驻角色：

- `Research Director`
- `State Keeper`
- `Judge`
- `Benchmark & Data Steward`

当前 phase 重点角色：

- `Feature Signal Auditor`
- `Learner Sensitivity Auditor`
- `Slice Constructor Owner`

下游影子角色：

- `Preference & Objective Owner`
- `Candidate/Search Owner`
- `Materialization & Validation Owner`
- `Surrogate Analyst`

触发型角色：

- `Literature Radar`
- `Reproduction Engineer`

`v2` 推荐主拓扑：

- `Orchestrator`
  负责分解目标、预算、调度、综合，不直接做大段实现
- `Proposer`
  负责形成 design card / experiment card
- `Reviewer`
  负责反对、挑错、攻击假设与风险
- `Arbiter`
  负责在 proposer 与 reviewer 之间作出 `approve / revise / reject`
- `Worker`
  负责在隔离上下文或 worktree 中执行被批准的任务
- `Verifier`
  负责测试、artifact 汇总、run manifest、机械检查
- `State Keeper`
  负责把事实写回 `.slicetune`

### 自动科研循环规则

- 单轮只改一个主变量
- 固定 benchmark / baseline / 指标层级 / 预算等级
- 先跑 `Tier A / Tier B`，再决定是否升级到 `Tier C`
- 失败实验必须保留并解释

一个正式 loop 至少包含：

1. `ExperimentCard`
2. `RunManifest`
3. `ResultBundle`
4. `JudgeReport`
5. `DecisionRecord`

### Debate 先于执行

以下事项默认必须先走 debate，再进入执行：

- 新 feature family
- 新 clustering principle
- 新 objective
- 新 search algorithm
- 新 learner / benchmark
- 新论文方法接入主线

标准 debate 顺序：

1. `DesignCard`
2. `ReviewCard`
3. `RevisedDesign` 或补充说明
4. `DebateDecision`
5. 只有 `approve` 后才能进入 experiment card 或代码实现

强规则：

- proposer 不能给自己当 reviewer
- reviewer 不能直接改 proposer 原稿后宣布通过
- arbiter 默认不重做设计，只做裁决
- 非 trivial 决策没有 review artifact 时，不允许 promote

### Champion-Challenger

- `Champion`：当前主线基线
- `Challenger`：新候选

替换条件：

- challenger 在固定口径下稳定优于 champion
- 且改善超过 noise floor，或满足预设局部目标

### Phase-locked execution

自动循环默认只能在当前 phase 中工作。

如果当前 phase 是 `Feature Signal Audit`，则 Generator 只能优先提出：

- noise floor
- learner sensitivity
- feature intervention
- feature-related benchmark refinement

不能默认把主资源投到：

- surrogate family 搜索
- 更复杂 beam policy
- UI polishing

### Literature Radar

触发条件：

- 同一问题连续多轮未超过 noise floor
- 当前 learner 明显不对数据组成敏感
- 当前 heuristic 无 defendable principle
- clustering / candidate / materialization 进入明显瓶颈

重点信息源：

- arXiv
- OpenReview
- CVF Open Access
- ACL Anthology
- PMLR
- IEEE / ACM 正式页面
- Papers with Code
- 开源仓库 README / docs / code

每次至少输出：

1. 一个 `landscape_brief`
2. `3-5` 个 `method_card`
3. 一个最小复现优先级建议

### 当前最适合自动化的 loop

最适合先自动化的是：

1. `Noise Floor Loop`
2. `Learner Sensitivity Loop`
3. `Feature Intervention Loop`
4. `Slice Leverage Loop`

当前不应优先自动化：

- surrogate 大搜索
- objective 任意扩张
- 无约束 candidate generation

### Judge Policy 与 Run Manifest

为了防止 proposer 直接调自己的过关标准：

- judge threshold 不应默认写在 experiment card 中
- experiment card 默认只引用独立的 `judge_policy`
- `judge_policy` 视为 reviewer / judge 侧控制对象

每次正式 tick 都应自动写出 `run_manifest.json`，至少记录：

- experiment id
- card path
- judge policy path
- git SHA / branch / dirty state
- python / platform / hostname
- invoked command
- started / finished time
- output dir

只有 `result bundle + judge report + run manifest` 同时存在时，结果才算具备基本可审计性
