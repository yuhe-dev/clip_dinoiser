# 决策日志

更新时间：2026-04-15

---

## DEC-001

- 决策：正式建立 `AGENTS.md + .slicetune/` 作为本项目本地研究记忆与运行骨架
- 结论：`keep`
- 原因：
  - 当前项目已具备多阶段 pipeline，但跨会话状态尚未显式持久化
  - 需要把长期原则、当前 phase、任务队列、决策日志分层管理

---

## DEC-002

- 决策：当前项目主线阶段锁定为 `Phase 1：Feature Signal Audit / Learner Sensitivity Audit`
- 结论：`keep`
- 原因：
  - 当前全局结果动态范围较窄
  - 下游模块复杂度已经高于可 defend 证据密度
  - 必须先确认 response signal 是否存在以及来自哪里

---

## DEC-003

- 决策：将 `surrogate refinement` 与 `candidate generation` 扩张从主线中暂时下调
- 结论：`park`
- 原因：
  - 上游 feature / learner / slice leverage 尚未形成强证据
  - 继续强化下游复杂模块风险较高

---

## DEC-004

- 决策：设立 `Literature Radar` 为触发型角色，而不是常驻主线 owner
- 结论：`keep`
- 原因：
  - 文献调研在瓶颈期非常关键
  - 但如果常驻接管主线，容易提前引入过多替代分支

---

## DEC-005

- 决策：当前所有规范文件先采用中文审阅版
- 结论：`keep`
- 原因：
  - 便于先审阅结构、职责边界和术语
  - 内容确认后再统一切换英文正式版

---

## DEC-006

- 决策：在研究规范之外，单独建立工程规范与代码结构契约
- 结论：`keep`
- 原因：
  - 当前仓库存在顶层脚本较多、逻辑分散、阶段性快写代码遗留的问题
  - 后续若要让 agent 持续自动迭代，必须先明确模块边界、脚本职责与重构方向

---

## DEC-007

- 决策：后续新增功能默认优先进入包层或后续 `research_harness/`，不再继续扩张根目录临时脚本
- 结论：`keep`
- 原因：
  - 需要抑制工程债进一步增长
  - 需要为自动执行、测试与复用创造结构基础

---

## DEC-008

- 决策：参考 `autoresearch` 的小步快跑思想，但为本项目补充 phase gate、benchmark contract 与证据晋升规则
- 结论：`keep`
- 原因：
  - 通用自动科研循环强调窄修改面与快速筛选，这一点适合本项目
  - 但本项目训练成本更高、pipeline 更长、噪声问题更强，因此必须增加更严格的治理层

---

## DEC-009

- 决策：自动科研系统默认采用 Champion-Challenger、Tiered Budget、Negative Result Logging 三项机制
- 结论：`keep`
- 原因：
  - 可减少高成本误跑
  - 可避免单次最优误导主线
  - 可迫使系统保留失败证据，而不是只保留看起来“成功”的轨迹

---

## DEC-010

- 决策：继续把工程规范下沉到包级开发契约，而不是只停留在抽象“高内聚低耦合”口号
- 结论：`keep`
- 原因：
  - 当前仓库已经形成 `feature_utils`、`slice_discovery`、`slice_remix` 三个核心包
  - 若不把边界进一步写细，后续重构仍会缺少落地方向

---

## DEC-011

- 决策：将 `testing_and_validation_contract` 与 `review_and_change_control_contract` 单独建档
- 结论：`keep`
- 原因：
  - 研究仓库中“代码验证”和“研究结论验证”容易混淆
  - 需要单独约束何时改 benchmark、何时补测试、何时允许 promote

---

## DEC-012

- 决策：为 Literature Radar 建立独立契约，并将其作为触发型角色而不是常驻随意发散源
- 结论：`keep`
- 原因：
  - 你明确希望在瓶颈时优先访问顶会文献与开源仓库
  - 但若缺少触发条件与输出格式，Radar 很容易引入过多无序分支

---

## DEC-013

- 决策：将 `.slicetune/context/` 与 `.slicetune/state/` 的大量细分文件回收为少量主文件
- 结论：`keep`
- 原因：
  - 过细粒度的文件拆分会降低可读性与维护效率
  - 更适合当前阶段的方式是：少量主文件承载规则，模板与日志保留细分

---

## DEC-014

- 决策：进一步将规则层收敛为 `program.md + playbook.md`，状态层收敛为 `board.md + decision_log.md`
- 结论：`keep`
- 原因：
  - 当前仓库更需要高频可读的主入口，而不是继续增加“概念上很清晰、实际阅读很碎”的文件
  - 对当前阶段而言，研究契约、工程规范、自动运行规则更适合合并到一个执行手册中
  - 当前任务、实验队列、分支状态、重构待办更适合合并到一个执行面总表中

---

## DEC-015

- 决策：`research_harness` 采用“薄执行层 + agent 主导决策”的实现方式，而不是把科研判断全部硬编码到脚本里
- 结论：`keep`
- 原因：
  - agent 负责提出假设、切换 phase、解释结果、触发 Literature Radar
  - harness 只负责稳定执行、结构化结果、固定 judge 规则和状态回写所需的最小产物
  - 这样既保留 agent 的科研灵活性，又保证后台循环可重复、可恢复、可审计

---

## DEC-016

- 决策：以 `EXP-P1-001` 作为第一条最小可执行 loop，先自动汇总现有 noise floor，再决定下一条 loop
- 结论：`keep`
- 原因：
  - 该 loop 不需要新训练成本，却直接决定后续 promote / park / kill 的地板线
  - 实际运行结果显示：`192` 个 `1000` 图随机子集的 global `mIoU` `mean=24.2939`、`std=0.0260`、`range=0.14`
  - 这说明当前全局指标动态范围确实很窄，下一步应优先推进 `EXP-P1-002` 和 `EXP-P1-003`

---

## DEC-017

- 决策：长时 autonomous research loop 采用 `Orchestrator / Proposer / Reviewer / Arbiter / Worker / Verifier / State Keeper` 的 v2 主拓扑
- 结论：`keep`
- 原因：
  - 官方 Anthropic / OpenAI 材料都强调长任务需要多角色分工、独立评审和结构化 handoff
  - 当前系统最缺的是 reviewer 与 arbiter 的强制 gate，而不是再增加更多无边界 agent

---

## DEC-018

- 决策：judge threshold 从 experiment card 中外置到独立 `judge_policy`，并要求每次正式 tick 自动写出 `run_manifest.json`
- 结论：`keep`
- 原因：
  - 这样可以降低 proposer 直接调自己过关标准的风险
  - run manifest 可把 git / 环境 /命令 /时序等可审计信息冻结下来，提升科研可复现性与可恢复性

---

## DEC-019

- 决策：执行逻辑中的关键 gate 默认由代码控制，而不是只由 prompt 约束
- 结论：`keep`
- 原因：
  - debate、phase、human review 这类关键约束若只存在于 prompt，长时运行时容易漂移
  - 因此新增 `run_research_queue.py + controller_policy.json`，由 controller 显式阻止未通过 gate 的任务继续执行

---

## DEC-020

- 决策：任何被标记为 `requires_debate` 的非 trivial 任务，必须在代码层通过 debate gate；任何被标记为 `human_review_required` 或 `phase_completion_candidate` 的任务，成功后默认自动停止并等待人工审核
- 结论：`keep`
- 原因：
  - 这符合“重要设计先 debate、阶段完成后必须人类审核”的目标
  - 代码级 stop gate 比聊天提醒更适合数小时连续运行

---

## DEC-021

- 决策：worker runtime 从裸 `python` 路径升级为代码注册的 runtime profile，并在 controller 中引入 preflight gate
- 结论：`keep`
- 原因：
  - `EXP-P1-002` 首次真实运行直接暴露出 base env 缺少 `torchvision / mmcv`
  - 若不把 runtime 选择做成代码约束，agent 长时运行会不断把错误环境误当成可用 worker
  - 因此新增 `.slicetune/runtime/runtime_profiles.json`，并由 controller 在执行前选择 `clipdino2 / clipdino / current_python`

---

## DEC-022

- 决策：多 seed 任务的 resume 不再只依赖 `result.json` 是否存在，而改为 `result.json + completion.json` 双条件
- 结论：`keep`
- 原因：
  - 只凭结果文件存在会把半截结果、旧结果、坏结果误判为已完成
  - 长时自治系统必须能区分“真正完成的旧 run”和“中途中断的残留目录”
  - 因此在 `research_harness/multi_seed.py` 中新增 completion sentinel，并在异常时把卡片原子落到 `failed_execution`

---

## DEC-023

- 决策：lease 与 stale reclaim 补充进程身份信息，并在 reclaim 前检查本机 pid 存活
- 结论：`keep`
- 原因：
  - 只靠时间戳回收运行卡片，容易误回收仍活着但短暂失去 heartbeat 的进程
  - 当前至少需要记录 `pid / hostname / runtime_profile_id` 并做本机存活校验，降低长任务状态污染风险

---

## DEC-024

- 决策：为每个实验 loop 输出统一的 `task_plan.json + progress.md`，把执行从“卡片级”向“任务级 conductor”推进
- 结论：`keep`
- 原因：
  - PM_agent 方法论强调 task 粒度增量执行和干净上下文交接
  - 仅靠 `progress.json` 不够支持 task 级 handoff 与 review
  - `task_plan.json + progress.md` 可作为后续 debate / reviewer / human acceptance 的统一落点

---

## DEC-025

- 决策：debate gate 从“仅检查字段”升级为独立的 `debate` 验证模块，并支持可选的 artifact 追踪校验
- 结论：`keep`
- 原因：
  - 需要让 debate 成为真正的可审计门禁，而不是口头约定

## DEC-026

- 决策：将 `learner_sensitivity_ladder` 升级为 Phase 1 主线中的正式可执行 loop，并允许其自动接入 proposer / debate / agentic judge 主闭环
- 结论：`keep`
- 原因：
  - 只有真实跨 regime 运行，才能区分“模型天然不敏感”和“当前训练协议选择不合适”
  - `EXP-P1-003` 的真实运行结果已经证明，这一 loop 不只是设计态 artifact，而是有效的证据生成器

## DEC-027

- 决策：保持 `feature_experiment_fast_cached_slide` 作为当前 Phase 1 的基线 learner regime，并将 `feature_intervention_matrix` 作为下一条主线实验
- 结论：`keep`
- 原因：
  - `EXP-P1-003` 中 `fast_cached_1ep=24.29`，明显高于 `fast_1ep=20.39` 与 `standard_3ep=20.75`
  - 当前更紧迫的问题不再是“换哪个训练 regime”，而是“在最稳定可用的 regime 上审计 feature 是否真的带来可重复响应”

## DEC-028

- 决策：将 `EXP-P1-003` 的解释从“learner sensitivity 已解决”修正为“protocol sensitivity established”，并在 `feature_intervention_matrix` 前插入 `learner adaptability audit`
- 结论：`keep`
- 原因：
  - `EXP-P1-003` 一次同时改变了 model/config bundle、effective budget 以及 eval path，控制变量仍不够干净
  - 当前代码只训练 `obj_proj + bkg_decoder`，而 `clip_backbone / FOUND / DINO` 全冻结，说明 learner 可能并非“太强”，而是“太僵”
  - 因此当前更合理的问题不再是“换哪个 regime 分更高”，而是“扩大训练算法的可塑性后，数据组成响应是否会显著放大”

## DEC-029

- 决策：`learner adaptability audit` 的第一轮实验采用 `少量 probe feature axes + real/shuffled/random 对照 + per-learner noise floor` 的最小设计，而不是直接使用整个特征空间或做大搜索
- 结论：`keep`
- 原因：
  - 当前有两个不确定变量：`learner algorithm` 与 `feature definition`
  - 若同时放开大搜索，结果会失去可解释性
  - 现有 `quality / difficulty / coverage` feature space 应先被当作 `working hypothesis / probe axes`，而不是默认真理
  - `shuffled/random` 对照能帮助区分“feature 真有信号”与“只是 materialization 或随机扰动带来的差异”
  - 为后续引入 `design_card / review_card / arbiter_decision` 自动化打底

---

## DEC-026

- 决策：增加 `run_research_debate.py`，用于 debate bundle 的组装与校验
- 结论：`keep`
- 原因：
  - 让 debate 输出具备可复现的生成/校验入口
  - 为后续自动化 debate pipeline 提供稳定 CLI 入口

---

## DEC-027

- 决策：controller 增加 task-level acceptance gate（基于 `task_plan.json` 的 `acceptance_status`）
- 结论：`keep`
- 原因：
  - 让 task 级验收真正成为自动停机点
  - 逐步接近 PM_agent 中的 task 级 review / acceptance 机制

---

## DEC-028

- 决策：新增 task board（`task_board.json`）用于跨实验汇总 task-level 状态，并由 daemon 自动更新
- 结论：`keep`
- 原因：
  - 需要一个跨实验的任务级控制面
  - 便于 conductor 发现“哪些任务卡在等待人审或阻塞状态”

---

## DEC-029

- 决策：采用研究版 task 状态机（`hypothesis → design → audit → execution → verification → judgment → acceptance`）
- 结论：`keep`
- 原因：
  - 与当前 Phase 1 的科研任务更一致
  - 与 task-level progress / handoff 的设计自然对齐

---

## DEC-024

- 决策：scheduler 从“按文件名顺序找 queued 卡”升级为“显式 priority + depends_on + readiness”模型
- 结论：`keep`
- 原因：
  - 自动科研循环不能继续依赖文件名排序，否则容易让下游卡片抢跑上游研究 gate
  - 因此新增 `research_harness/scheduler.py`，并把 queue snapshot 也纳入机器可读状态

---

## DEC-025

- 决策：每次正式执行都要记录为独立 attempt，并为 attempt 保留独立 manifest 与 artifact snapshot
- 结论：`keep`
- 原因：
  - card 记录“实验是什么”，attempt 记录“这次具体怎么跑的”
  - 如果没有 attempt 维度，重试、恢复、环境切换和失败证据都会混成一团

---

## DEC-026

- 决策：引入 daemon loop 作为长时间自治运行入口，但默认保留 human review stop 和 fatal failure stop
- 结论：`keep`
- 原因：
  - 用户目标不是单次实验自动化，而是数小时连续推进
  - 但若没有 stop gate，daemon 很容易越权推进阶段或持续消耗错误卡片

---

## DEC-027

- 决策：新增 `max_attempts` 与 `blocked_retry_limit`，作为 retry 上限保护
- 结论：`keep`
- 原因：
  - 长时间自治系统不能对失败卡无限重试
  - 到达 retry 上限后，应默认阻断并等待人类或上层 agent 明确调整

---

## DEC-028

- 决策：新增 `runtime_index` 吸收层，把 cards、attempts、judge reports、run manifests 聚合成机器可读全局状态
- 结论：`keep`
- 原因：
  - daemon、后续 agent 和人类验收都需要一个稳定的全局状态视图
  - markdown `board.md` 适合人读，但不适合作为持续自动循环的唯一状态入口

---

## DEC-029

- 决策：新增 phase-locked proposer 层，但默认只允许按 `proposal_policy` 生成 follow-on proposal，不允许自由发明新主线
- 结论：`keep`
- 原因：
  - 用户希望系统能在长时运行中自动决定下一步
  - 但当前阶段不能让 proposer 擅自扩张研究分支
  - 因此 proposer 只消费 `runtime_index + proposal_policy`，并优先生成 `draft_only` / `planned` 提案

---

## DEC-030

- 决策：显式区分 `design_only` 与 `executable` card，taskflow 只自动推进具备 runtime handler 的 loop
- 结论：`keep`
- 原因：
  - auto-proposer materialize 出来的计划卡不应被 daemon 误当成可执行实验
  - 成熟 conductor 必须把“研究设计”和“实验执行”在代码层分开

---

## DEC-031

- 决策：debate bundle 统一支持 legacy/new schema，并允许 daemon 在 `planned / blocked_debate_gate` 上自动生成 debate 产物
- 结论：`keep`
- 原因：
  - 长时自治不能要求每张设计卡都手工补 debate 文件
  - 同时必须兼容已有 debate 资产，避免 schema 切换时状态污染

---

## DEC-032

- 决策：为每个 queue session 写出 `context_packet.json`，把 card、task、runtime、repo memory 压缩成最小可恢复上下文
- 结论：`keep`
- 原因：
  - 用户明确要求把 context 当作程序内存精细操控
  - `memory_paths` 本身不足以支撑真正的跨会话恢复与 reviewer 接手

---

## DEC-033

- 决策：增加 human approval release 机制，允许 daemon 在人工批准后自动把 `awaiting_human_review` 卡片释放到 `completed / queued`
- 结论：`keep`
- 原因：
  - 否则 human review 只是单向停机，无法形成真正闭环
  - 这让“人类裁决阶段停机，但后续推进由系统接手”第一次成为代码级状态机

---

## DEC-034

- 决策：queue runner 增加可选 tick watchdog；开启 `tick_timeout_seconds` 后，tick 改为子进程执行并可在超时后强制失败收口
- 结论：`keep`
- 原因：
  - 这是长时 autonomous system 避免单次 hang 永久续租的关键保险
  - 默认保持关闭以兼容现有 loop，但执行层已具备 watchdog 能力

---

## DEC-035

- 决策：dynamic literature radar 只对研究级 retry exhaustion 触发，不对纯 harness 故障触发
- 结论：`keep`
- 原因：
  - 基础设施 bug 不应被误判为算法瓶颈
  - 当前将触发条件收严为 `blocked_retry_limit`

---

## DEC-036

- 决策：将 `EXP-P1-002` 的固定 subset 多 seed 结果正式记为 Phase 1 的第二条有效基线
- 结论：`keep`
- 原因：
  - 当前结果：`mean=24.2860`、`std=0.0089`、`range=0.0200`
  - 该 training noise 显著低于全局 random-subset floor `std=0.0260`
  - 说明当前数据组成效应仍值得继续审计，而不应把全部窄幅波动都归因于训练噪声

---

## DEC-037

- 决策：将 agentic artifact 层正式接入 queue/tick/daemon 主闭环，而不再只保留独立 CLI 入口
- 结论：`keep`
- 原因：
  - 之前 `agentic` 层虽然存在，但没有进入真实 autonomous loop
  - 现在正式将 `hypothesis_brief / design_pack / evaluation_rubric / analysis_brief` 接入 tick 前后与 daemon 巡检

---

## DEC-038

- 决策：将 `literature_radar` 从 design-only proposal 升级为真实可执行 loop
- 结论：`keep`
- 原因：
  - 文献雷达如果只会立项，不会检索，就无法真正扩展研究空间
  - 现在已支持 query planning、OpenAlex retrieval、ranking、method cards 与 human-review stop

---

## DEC-039

- 决策：phase-locked proposal 若未显式给出 `output_dir / debate_bundle_path`，由 proposer 自动补默认路径
- 结论：`keep`
- 原因：
  - 真实 daemon 运行暴露出 `EXP-P1-003` 虽已 materialize，但缺少稳定 artifact 根目录与 debate 入口
  - proposal 作为执行合同，不能把这些关键路径留空

---

## DEC-040

- 决策：completed card 的 task plan 应允许被 taskflow 回收重建，以消除 release 后的 stale acceptance 状态
- 结论：`keep`
- 原因：
  - 人类审核通过后，card 可能已从 `awaiting_human_review` 释放为 `completed`
  - 若仍保留旧 `task_plan.json`，会出现“card 已完成但 task 仍在等验收”的状态污染

---

## DEC-041

- 决策：judge 升级为“两层制”，即 mechanical judge + agentic rubric/context overlay judge
- 结论：`keep`
- 原因：
  - 纯硬规则 judge 不足以承载不同 loop 的研究解释
  - 当前已让 planner 先冻结 `evaluation_rubric`，再由 `agentic_judge` 结合 rubric、context packet、result bundle 和机械 judge 生成最终 `judgment_brief`

---

## DEC-042

- 决策：将 `EXP-P1-002` 的人工审核结论正式写入 `human_review.json`，并允许 daemon 自动释放为 `completed`
- 结论：`keep`
- 原因：
  - 用户已明确口头认可该结果
  - 释放后系统成功自动 materialize 了 `EXP-P1-003`，证明 human stop + release 闭环已真正跑通

---

## DEC-043

- 决策：将 `EXP-P1-004` 冻结为“最小 learner adaptability audit”，而不是立即执行泛化版 `feature_intervention_matrix`
- 结论：`keep`
- 原因：
  - `EXP-P1-003` 更适合解释为 `protocol sensitivity audit`
  - 当前真正要回答的是：扩大 learner 可训练范围后，真实 feature-guided intervention 是否会出现更强响应
  - 因此需要先固定：
    - `L0/L1/L2` 三个 learner variants
    - `quality_sharpness / difficulty_small_object` 两条 probe axes
    - `real / shuffled / matched-random` 三类 controls
    - learner-specific noise floor

---

## DEC-044

- 决策：当前 feature space 在 `EXP-P1-004` 中只被视为 `working hypothesis probe axes`，不是 ground truth
- 结论：`keep`
- 原因：
  - 当前 `learner algorithm` 与 `feature definition` 同时不确定，不能两边一起做大搜索
  - 更稳妥的做法是：
    - 先用少量现有 feature axes 作为 probe
    - 配 `shuffled / matched-random` 对照
    - 记录 realized intervention fidelity
  - 这样才能区分：
    - learner 太僵
    - feature 无效
    - learner–feature coupling 确实存在

---

## DEC-045

- 决策：`agentic planner` 必须能从 `experiment card metadata` 稳定编译出阶段特化的 planning artifact，而不是反复回退到 generic 模板
- 结论：`keep`
- 原因：
  - `EXP-P1-004` 的细化设计如果只存在于手工 artifact，会被 daemon 下一次回填覆盖
  - 因此已将 `minimal_learner_adaptability_audit` 接入 `planner.py`
  - 现在 `design_mode + metadata` 已成为 `hypothesis_brief / design_pack / evaluation_rubric` 的稳定上游合同

---

## DEC-046

- 决策：将 `EXP-P1-004` 的 `Tier A` 先落成最小 executable runtime，而不是一次性接通全部 controls / tiers
- 结论：`keep`
- 原因：
  - 当前主目标是尽快建立真实 learner adaptability evidence，而不是先把所有 control families 一起做复杂
  - 因此先只接：
    - `L0/L1/L2`
    - `quality_sharpness / difficulty_small_object`
    - `real_feature_guided`
    - learner-specific noise floor
  - 这能让 Phase 1 尽快从“设计态”进入“真实运行态”

---

## DEC-047

- 决策：为 learner adaptability 审计新增 selective-unfreeze 训练能力，并将其作为训练脚本的显式参数
- 结论：`keep`
- 原因：
  - 当前 learner sensitivity 的核心问题不再只是 config bundle，而是 learner 可训练范围
  - 因此新增：
    - `helpers/trainability.py`
    - `run_remix_training_experiment.py --trainable-modules`
    - `feature_experiment_pipeline.py` 中的 trainable-scope 配置
  - 这样 `L1/L2` 才能在不改 teacher 的前提下形成受控的 learner adaptability ladder

---

## DEC-048

- 决策：修复 runtime preflight 对 `clipdino2` 的 false negative，并通过重启 daemon 让 `EXP-P1-004` 进入真实执行
- 结论：`keep`
- 原因：
  - 实机确认机器具备 4 张可见 GPU，`clipdino2` 在正常导入顺序下可正确识别 CUDA
  - preflight 失败来自 `MKL_THREADING_LAYER` / 导入顺序导致的探针误判，而不是机器真的无 GPU
  - 因此将 probe 调整为：
    - 先导入 `numpy`
    - 设置 `MKL_SERVICE_FORCE_INTEL=1`
    - 设置 `MKL_THREADING_LAYER=GNU`
  - 修复后，`EXP-P1-004` 已在新版 daemon 下进入 `running`

---

## DEC-049

- 决策：保持 `clipdino2` worker 路径对 Python 3.9 兼容，不在训练执行链路中使用未延迟求值的 `|` 联合类型注解
- 结论：`keep`
- 原因：
  - `EXP-P1-004` 第一条真实训练在 worker 进程中失败，不是算法问题，而是 `feature_experiment_pipeline.py` 在 Python 3.9 下解析 `list[str] | None` 时直接报错
  - 这说明 research runtime 的系统 Python 与训练 worker Python 版本并不一致，训练链路必须优先保证低版本兼容
  - 修复后，worker 已能继续向前推进到真正的训练逻辑

---

## DEC-050

- 决策：向 Hydra `cfg.train` 注入 `trainable_modules` 时必须通过 `open_dict` 显式打开 struct 写入
- 结论：`keep`
- 原因：
  - `EXP-P1-004` 第二次真实训练失败来自 OmegaConf/Hydra 的 struct 保护，而不是 selective-unfreeze 设计本身错误
  - 若不先修正配置注入方式，后续任何 learner adaptability 变体都无法真正进入训练
  - 修复后，`EXP-P1-004` 已进入 `L0_head_only_noise_seed00` 的真实训练阶段

---

## DEC-051

- 决策：将 `EXP-P1-004` 首个 `L0_head_only_noise_seed00` 结果解释为“runtime fidelity check passed”，而不是“learner adaptability 已有科学结论”
- 结论：`keep`
- 原因：
  - 首个真实结果已落盘：`mIoU=24.29`
  - 该数值与先前固定 subset、同配置的 head-only 基线一致
  - 这首先说明新接入的 `feature_intervention_matrix + selective-unfreeze` 执行链路没有明显破坏基线可比性
  - 但当前只完成了一个 `L0` noise seed，尚不足以计算 learner-specific noise floor，更不足以比较 `L0/L1/L2` 对 probe feature axes 的响应差异

---

## DEC-052

- 决策：将 `EXP-P1-004` 的 `Tier A` 最终解释为“当前轻量 learner ladder 未能区分 composition sensitivity”，而不是“learner adaptability 成功建立”
- 结论：`keep`
- 原因：
  - `EXP-P1-004` 已正式完成并写出 `result_bundle / judge_report / analysis_brief / judgment_brief`
  - 三个 learner variants 的 noise floor 完全一致：
    - `L0=24.29`
    - `L1=24.29`
    - `L2=24.29`
    - `std≈0`
  - 两条 probe axes 的 response 也在三个 learner 上完全一致：
    - `quality_sharpness` amplitude=`0.15`
    - `difficulty_small_object` amplitude=`0.13`
  - 因此当前通过增加 `decode_head.proj`、最后一个 CLIP visual block、`ln_post` 的 trainable scope，并没有放大数据组成响应
  - 这条结论比“Tier A 机械 screen 通过”更重要，因为它直接说明当前 ladder 的科学区分度不够

---

## DEC-053

- 决策：将 `EXP-P1-004` 正式记为 `park`，并把下一步分成两条互斥优先方向，而不是继续默认沿当前 ladder 扩张
- 结论：`keep`
- 原因：
  - 当前 mechanical summary 显示：
    - `real_cells_above_noise_floor_count=6`
    - `screen_passed=true`
    - `mean_off_target_drift_ratio=0.0154`
  - 但 frozen rubric 未通过：
    - `real_beats_shuffled=false`
    - `real_beats_random=false`
    - 当前只执行了 `real_feature_guided`
  - 因此当前最合理的后续不是直接宣称 feature/learner 成立，而是二选一：
    - 路线 A：推进 `Tier B` controls，验证现有 feature axes 是否真的优于 shuffled/random
    - 路线 B：先扩 learner 分支，寻找比当前 `L0/L1/L2` 更强的 learner adaptability 差异
  - 在没有新证据前，当前 `L0/L1/L2` 这组三档轻量 ladder 应视为“已审计但未拉开差异”，不再默认继续细化同一路径

---

## DEC-054

- 决策：将当前 `L1/L2` 负结果重新解释为“现有 selective-unfreeze 设计可能未真正进入有效梯度路径”，而不直接解释为“更可塑 learner 也没用”
- 结论：`keep`
- 原因：
  - 当前默认配置使用 `feats_idx=-3`，见 `configs/feature_experiment_fast_cached_slide.yaml`
  - `CLIP_DINOiser` 在训练时从 `resblocks[-3].ln_2` 注册 hook，并且将该特征 `detach()` 后存入 `train_feats['clip_inter']`
  - 训练 loss 中真正参与 `obj_proj / bkg_decoder` 计算的是这个 `detach()` 后的 `clip_inter`
  - 因此当前新增的：
    - `clip_backbone.decode_head.proj`
    - `clip_backbone.backbone.visual.transformer.resblocks.-1`
    - `clip_backbone.backbone.visual.ln_post`
    很可能虽然被设为 `requires_grad=True`，但并没有形成对当前 loss 的有效梯度贡献
  - 这意味着 `EXP-P1-004` 当前更像：
    - 一次成功的 runtime / materialization / feature-pair screen
    - 但不是一次足够干净的 learner adaptability causal test

---

## DEC-055

- 决策：为 learner adaptability 新增一条“真正接入 CLIP backbone 梯度”的审计分支，并保持原始默认训练逻辑不变
- 结论：`keep`
- 原因：
  - 新增 `feature_experiment_fast_cached_slide_backbone_grad` 配置：
    - `feats_idx=final`
    - `enable_clip_grad_for_training=true`
    - `detach_intermediate_train_feats=false`
  - `DinoCLIP.get_clip_features` 和 `MaskClip.extract_feat/forward` 现在支持显式 `track_grad`
  - 修复了 `MaskClip.extract_v` 中原先只适用于 `no_grad` 路径的原地加法
  - 通过真实 smoke test 已确认以下模块收到非零梯度：
    - `obj_proj`
    - `bkg_decoder`
    - `clip_backbone.decode_head.proj`
    - `clip_backbone.backbone.visual.transformer.resblocks.-1`
    - `clip_backbone.backbone.visual.ln_post`
  - 这说明我们现在终于有了一条“新增 CLIP 模块真的参与当前 loss 优化”的实验路径

---

## DEC-056

- 决策：将用户手工运行的第一次 backbone-grad 实验解释为“暴露了 `self.training` 门禁 bug 的排错 run”，而不是正式科学结果
- 结论：`keep`
- 原因：
  - 真实终端输出中：
    - `clip_backbone.decode_head.proj: params_with_grad=0/1`
    - `resblocks.-1: params_with_grad=0/12`
    - `ln_post: params_with_grad=0/2`
  - 问题原因不是 backbone-grad 思路错误，而是：
    - `set_train_mode_for_modules()` 会先 `model.eval()`
    - 原实现用 `self.training and enable_clip_grad_for_training` 来控制 `track_grad`
    - 导致在真实训练里 backbone-grad 被错误关掉
  - 现已修复为：
    - `enable_clip_grad_for_training and torch.is_grad_enabled()`
  - 并在与真实训练一致的 smoke test 下重新确认：
    - `decode_head.proj / resblocks.-1 / ln_post` 均收到非零梯度
  - 因此第一次手工 run 得到的 `mIoU=23.61` 不能被解释为“真正的 backbone-grad learner 结果”，更像一次有效的系统排错样本

---

## DEC-057

- 决策：将第二次 backbone-grad 手工 run 解释为“真实 backbone-grad 路径已生效，但当前 `L2 + final features` 组合在 anchor subset 上未带来正收益”
- 结论：`keep`
- 原因：
  - 第二次终端输出中，新增 CLIP 模块已明确收到非零梯度：
    - `decode_head.proj: params_with_grad=1/1, grad_norm=0.552573`
    - `resblocks.-1: params_with_grad=12/12, grad_norm=0.544799`
    - `ln_post: params_with_grad=2/2, grad_norm=0.028967`
  - 因此这次 run 可以被视为真正的 backbone-grad learner 实验
  - 该次结果为：
    - `mIoU=23.22`
    - 低于旧的 head-only anchor baseline `24.29`
  - 但当前还不能把这个差值直接解释成“放开 backbone 一定更差”，因为此次同时改变了：
    - `feats_idx: -3 -> final`
    - feature path: detached intermediate -> gradient-tracked final
    - learner scope: head-only -> L2 partial backbone
  - 所以当前最稳妥的解释是：
    - `stronger learner branch is now technically valid`
    - `this particular L2 + final-feature variant did not immediately improve the anchor subset`
    - 但还没有完成对 `L0/L1/L2` 的同配置因果比较

---

## DEC-058

- 决策：当前服务器的 4 张 GPU 优先用于“并行跑多个单卡实验”，而不是直接把当前 manual learner runs 改成 4-GPU DDP
- 结论：`keep`
- 原因：
  - 当前 `run_remix_training_experiment.py` 虽然会 `init_dist("pytorch")`，但训练阶段没有把 `model` 包进 `DistributedDataParallel/MMDistributedDataParallel`
  - 当前 `train_loader` 也没有使用 `DistributedSampler`
  - 因此如果直接把 `--nproc_per_node` 从 `1` 改成 `4`，训练很可能不是“正确同步的 4 卡训练”，而是“4 个进程各自乱跑同一数据”的错误模式
  - 在当前实现下，更安全也更高效的策略是：
    - GPU0 跑 `L0`
    - GPU1 跑 `L1`
    - GPU2 跑 `L2`
    - GPU3 跑另一条 high/low manifest 或第二个 seed
  - 等需要时，再单独实现真正的 DDP 训练路径

---

## DEC-059

- 决策：将“中间特征不 detach、并解冻 hook 上游 block”的方案正式定义为一条独立的 `intermediate-grad learner family`
- 结论：`keep`
- 原因：
  - 旧 family 使用：
    - `feats_idx=-3`
    - `detach_intermediate_train_feats=true`
    - 因此更像“固定中间特征 + 小头部学习”
  - 当前用户提出的方向不是继续扩 `final-feature backbone-grad`，而是：
    - 保持中间特征作为训练输入
    - 去掉 `detach`
    - 专门解冻能影响该中间特征的上游 CLIP blocks
  - 这与已建立的 `final-feature backbone-grad` 逻辑不同，应作为另一条 learner family 独立审计，而不是混在一起解释
  - 已新增配置：
    - `configs/feature_experiment_fast_cached_slide_intermediate_grad.yaml`
  - 该 family 的最小 ladder 定义为：
    - `L0`: `obj_proj + bkg_decoder`
    - `L1`: `L0 + clip_backbone.backbone.visual.transformer.resblocks.-3`
    - `L2`: `L1 + clip_backbone.backbone.visual.transformer.resblocks.-4`

---

## DEC-060

- 决策：将 `tools/run_backbone_grad_manual.py` 扩展为可生成 `final` 与 `intermediate` 两条 learner family 的手工命令
- 结论：`keep`
- 原因：
  - 当前 Phase 1 正在快速试验不同 learner family
  - 如果继续手写命令，容易把：
    - config family
    - variant trainable modules
    - GPU / port / output-dir
    混写出错
  - 现在脚本已支持：
    - `--family final`
    - `--family intermediate`
    - `--variant L0/L1/L2`
  - 这样用户可直接在终端复制命令，并且更容易做 4 卡并行

---

## DEC-061

- 决策：先以随机输入单步反传完成 `intermediate-grad` 的梯度烟雾审计，再允许用户大规模并行跑 `L0/L1/L2`
- 结论：`keep`
- 原因：
  - 之前的 `final-feature backbone-grad` 分支已经出现过“名义上解冻、实际上没进梯度路径”的问题
  - 因此新 family 必须先做最小梯度验证
  - 当前 smoke audit 已确认：
    - `obj_proj: 2/2`
    - `bkg_decoder: 2/2`
    - `clip_backbone.backbone.visual.transformer.resblocks.-3: 8/12`
    - `clip_backbone.backbone.visual.transformer.resblocks.-4: 12/12`
    收到非零梯度
  - 其中 `resblocks.-3` 不是 `12/12`，这反而符合预期：
    - 当前 hook 位于 `resblocks[-3].ln_2`
    - 因此 block 内 hook 之后的部分不一定在当前 loss 路径上
  - 这说明 `intermediate-grad` family 是一条技术上成立、值得继续跑 anchor baseline 的 learner 分支

---

## DEC-062

- 决策：将 `intermediate-grad` family 的首轮 anchor 结果解释为“技术路径成立，但当前在 CLIP-DINOiser 框架内未显现出更强 learner sensitivity”，并开始认真考虑引入更标准的 supervised segmentation probe learner
- 结论：`keep`
- 原因：
  - 用户已完成 4 个并行 anchor runs：
    - `L0 anchor seed0`: `mIoU=24.29`
    - `L1 anchor seed0`: `mIoU=24.20`
    - `L2 anchor seed0`: `mIoU=23.52`
    - `L2 anchor seed1`: `mIoU=23.52`
  - 当前解读：
    - `intermediate-grad` family 在技术上是成立的，因为 `resblocks.-3/-4` 已确认收到非零梯度
    - 但在 anchor subset 上，随着 trainable scope 增大，性能并未提升，反而下降
    - 且 `L2` 的两个 seed 结果几乎完全重合，说明当前这条 family 至少在 anchor baseline 上没有表现出“更活跃但更噪”的明显迹象
  - 这轮结果仍然不能直接回答“对 feature distribution 是否更敏感”，但它已经足够说明：
    - 当前继续围绕 CLIP-DINOiser 的 teacher-distillation learner family 深挖，未必是最高价值主线
  - 因此下一步值得认真考虑：
    - 引入一个更标准、更 data-sensitive 的 supervised segmentation learner 作为 `probe learner`
    - 先用它回答“feature-defined data composition 是否能稳定影响训练结果”
    - 再决定是否回到 CLIP-DINOiser / slice remix 主线做系统绑定

---

## DEC-063

- 决策：采纳“诊断型基础 learner”方向，但将其拆成两级，而不是只押单一路线
- 结论：`keep`
- 原因：
  - 另一条建议中提出的 `frozen dense feature + linear/MLP probe` 方向是有价值的，尤其适合回答：
    - 当前 feature axes 是否比 shuffled/random 更有信号
    - signal 是否已经存在于 frozen representation 中
  - 但它不能完全替代 supervised segmentation learner，因为它更接近：
    - frozen representation separability test
    - 而不是 image-space real training sensitivity test
  - 另外，当前仓库虽然有：
    - dense feature extraction 相关代码
    - visual embedding / coverage 资产
    - COCO-Stuff masks 与评测缓存
    但还没有现成的“dense feature probe training loop + mask downsample alignment”训练入口
  - 与之相比，当前环境已经现成安装了 MMSeg 0.27，并内置官方配置：
    - `deeplabv3plus_r50-d8_512x512_80k_ade20k.py`
    - `segformer_mit-b0_512x512_160k_ade20k.py`
  - 因此更合理的研究顺序是：
    1. 先保留 `frozen dense feature probe (P0/P1)` 作为便宜的 representation-level sanity test
    2. 但真正的主诊断 learner，优先采用 `DeepLabV3+ R50-D8`
    3. 如有必要，再用 `SegFormer MiT-B0` 做架构鲁棒性交叉验证

---

## DEC-064

- 决策：先将 `DeepLabV3+ R50-D8` 以“手动 supervised probe 训练脚本”的方式接入，而不是立刻并入现有 autoresearch queue
- 结论：`keep`
- 原因：
  - 当前目标是尽快得到一个真正 supervised、data-sensitive 的主 probe learner
  - 用户当前更希望先亲自在终端看到实际训练输出，而不是先做完整 queue 集成
  - 因此先新增：
    - `research_harness/supervised_probe.py`
    - `run_supervised_probe_experiment.py`
  - 这条脚本链路当前支持：
    - 读取 subset manifest
    - 构建 `DeepLabV3+ R50-D8` 的 MMSeg config
    - 过滤 COCO-Stuff train split 到 manifest 对应 basenames
    - 单卡 supervised training
    - full COCO-Stuff val evaluation
    - 写出 `result.json`
  - 现阶段先用它跑出第一批主 probe 结果，再决定是否并入 autoresearch queue 更合理

---

## DEC-065

- 决策：`DeepLabV3+ R50-D8` 的单卡手动 probe 路径默认改为 `BN`，不再沿用官方 config 的 `SyncBN`
- 结论：`keep`
- 原因：
  - 用户第一次手工运行 `run_supervised_probe_experiment.py` 时，训练在第一步 forward 失败：
    - `RuntimeError: Default process group has not been initialized`
  - 根因不是数据或模型结构，而是官方 MMSeg `DeepLabV3+ R50-D8` config 默认使用 `SyncBN`
  - 当前手动 supervised probe 入口是单卡、非分布式训练，不会初始化默认 process group，因此 `SyncBN` 必然报错
  - 已在 `research_harness/supervised_probe.py` 中加入递归配置改写，将 `cfg.model` 内全部 `SyncBN` 替换为普通 `BN`
  - 修复后的 `max_iters=1` smoke 已确认：
    - 训练成功进入真实 iter
    - 成功保存 checkpoint
    - 成功进入 full validation
  - 这说明当前主 probe learner 的单卡训练链路已经真实打通，后续用户可直接手工重跑正式 anchor 实验

---

## DEC-066

- 决策：将 `DeepLabV3+ R50-D8` 的第一次 `1000 iter` anchor 结果标记为“训练预算无效 / 不足以解释 scientific signal”，而不是负科学结论
- 结论：`keep`
- 原因：
  - 正式 anchor 运行已完成，结果为：
    - `mIoU=0.03`
    - `mAcc=0.60`
    - `aAcc=0.70`
  - 训练日志显示 loss 从约 `4.48` 降到约 `3.77`，说明模型不是完全没学，但明显远未收敛
  - 当前设置：
    - subset size = `1000`
    - `samples_per_gpu=2`
    - `max_iters=1000`
  - 这等价于只训练了约 `2` 个 epoch：
    - `1000 / (1000 / 2) = 2`
  - 对于带随机初始化 segmentation heads 的 `DeepLabV3+`，这远不足以形成可用 mIoU
  - 同时当前学习率和 schedule 直接继承自官方 `80k` iter config，仍属于“借用官方默认、但还未为 1000-image subset 重新校准”的状态
  - 因此这次 `0.03` 的意义是：
    - `DeepLab` 链路可以跑通
    - 但当前 supervised probe budget 尚未成立
    - 不能据此判断 feature 无效，也不能据此判断 learner 无 sensitivity

---

## DEC-067

- 决策：将表中外部 open-vocabulary baselines 按“当前仓库兼容性 + 代码可用性 + 对 Phase 1 的实际价值”分层，第一波优先级定为 `CLIP-DIY > MaskCLIP / MaskCLIP+ > TCL > GroupViT`，其余方法暂不作为首波接入对象
- 结论：`keep`
- 原因：
  - 当前用户希望参考 `CLIP-DINOiser` 论文表中的历史方法，筛选哪些值得在本仓库里实际试跑
  - 排序标准不是 leaderboard，而是：
    1. 是否有官方代码与可用权重
    2. 是否和当前仓库已有依赖/代码血缘接近
    3. 是否适合作为当前 Phase 1 的辅助 baseline，而不是把主线重新拖回复杂 open-vocabulary 预训练
  - 当前外部信息与本地仓库现实共同支持的结论：
    - `CLIP-DIY`：
      - 与 `CLIP-DINOiser` 同作者系谱，官方仓库公开
      - training-free
      - 直接依赖 FOUND saliency
      - 当前仓库已经安装/使用 FOUND，迁移成本最低
      - 适合作为“训练自由、同家族 predecessor baseline”
    - `MaskCLIP / MaskCLIP+`：
      - 官方代码公开
      - 当前仓库 `models/maskclip/` 已明确是从原始 MaskCLIP 修改而来
      - `MaskCLIP+` 需要 pseudo-label/self-training，但工程血缘最近
      - 适合作为“最容易复用的历史基线”
    - `TCL`：
      - 官方代码公开
      - 当前仓库 `main_eval.py` / `helpers/logger.py` 等有 TCL 血缘
      - 但其完整训练与评测链比 MaskCLIP 更重
      - 适合作为第二梯队
    - `GroupViT`：
      - 官方代码公开
      - 当前仓库 dataset/config 层已有 GroupViT 改写痕迹
      - 但环境版本较老、预训练依赖更重
      - 适合作为“可接但不应第一波”
    - `ReCo / NamedMask / OVDiff / OVSegmentor / SegCLIP`：
      - 均有官方代码
      - 但数据准备、环境版本、外部预计算/检索/扩散或专用预训练要求更重
      - 更适合作为“后续 robustness baseline”，而不是当前最先接的路线
    - `ZeroSeg / CLIPpy`：
      - 当前没有快速确认到清晰稳定的官方实现入口
      - 暂不建议作为首波对象
  - 因此下一步如果要在表内选一条“现在就试”的外部 baseline：
    1. `CLIP-DIY`
    2. `MaskCLIP`
    3. `MaskCLIP+`
    4. `TCL`
    5. `GroupViT`
  - 同时保留更高层判断：
    - 这些 open-vocabulary baselines 适合做辅助比较
    - 但当前 Phase 1 主诊断 learner 仍优先是 supervised probe（`DeepLabV3+ R50-D8`）

---

## DEC-068

- 决策：当前不因 `DeepLabV3+ R50-D8` 的低结果而立即切换数据集或切换任务；先将其解释为“supervised probe 接入仍存在 protocol bug”，修复后再判断 `COCO-Stuff / image segmentation` 是否需要降级
- 结论：`keep`
- 原因：
  - 用户完成了 `DeepLabV3+ R50-D8` 的两轮 anchor：
    - `1000 iter`: `mIoU=0.03`
    - `8000 iter`: `mIoU=0.08`
  - 这两个结果都低到远超“数据集动态范围小”所能解释的程度
  - 进一步排查发现：
    - `research_harness/supervised_probe.py` 借用了官方 `ADE20K` train pipeline
    - 其中 `LoadAnnotations` 默认 `reduce_zero_label=True`
    - 对 `COCO-Stuff164k` 而言这是错误的，因为其 train IDs 本来就是 `0..170`，`255` 才是 ignore
    - 这一点足以系统性破坏 supervision，使非常低的 mIoU 失去科学解释价值
  - 已修复为：
    - `LoadAnnotations(reduce_zero_label=False)`
  - 因此当前更合理的结论是：
    - supervised probe 结果暂时不能被解释为“COCO-Stuff 不行”
    - 也不能被解释为“image segmentation 这个任务不值得做”
  - 同时，结合 `CLIP-DINOiser` 论文表格本身可观察到：
    - 在 `COCO-Stuff` 列，不同方法仍存在明显差异（如 `13.3 -> 24.6`）
    - 在 `City`、`ADE` 列差异更大
    - 这说明 segmentation benchmark 本身并非没有算法动态范围
  - 因此当前优先级应为：
    1. 先在修复后的 supervised probe 上重新建立有效 anchor baseline
    2. 若仍异常，再继续排查学习率 / schedule / crop / subset budget
    3. 只有在“协议正确 + budget 合理”的前提下依然没有 signal，才讨论是否将 `COCO-Stuff` 降级或切换到更低熵 segmentation benchmark（如 `Cityscapes` / `ADE20K`）

---

## DEC-069

- 决策：将修复后 `DeepLabV3+ R50-D8` 的 `mIoU=1.11` 解释为“当前 supervised diagnostic protocol 对 `COCO-Stuff-171` 过于苛刻”，优先降级 benchmark 难度，而不是立刻放弃 `image segmentation` 任务
- 结论：`keep`
- 原因：
  - 修复 `reduce_zero_label=False` 后，训练日志已经明显比此前健康：
    - `decode.acc_seg` 常见于 `18%~26%`
    - `decode.loss_ce` 下降到约 `2.2~2.6`
  - 这说明当前 run 已不再是“完全跑坏”
  - 但 full val 结果仍然很低：
    - `mIoU=1.11`
    - `mAcc=2.64`
    - `aAcc=22.79`
  - 进一步拆 per-class 发现：
    - 只学会了少数容易的大类 / stuff 类
    - 仍有约 `150 / 171` 类 IoU 为 `0`
    - 有明显响应的主要是：
      - `sky-other`
      - `grass`
      - `snow`
      - `tree`
      - `wall-concrete`
      - `sea`
      - `person`
      - `road`
  - 这更像是：
    - `1000` 张随机子集
    - `171` 类高熵标签空间
    - full COCO-Stuff val
    - `ImageNet` 预训练 backbone + 随机初始化 segmentation heads
    的组合，对当前 supervised probe 过于严苛
  - 因而当前更合理的调整顺序是：
    1. 保留 `image segmentation` 任务
    2. 先降级 benchmark 难度
    3. 优先考虑与当前仓库数据资产最接近、熵更低的 segmentation benchmark
  - 推荐顺序：
    1. `COCO-Object`
    2. `VOC20`
    3. `Pascal Context59`
  - 不建议当前直接切换任务类型，因为：
    - 这会丢失当前仓库与 `CLIP-DINOiser` / `MaskCLIP` / `TCL` 等方法的直接可比性
    - 也会让当前 feature / slice 资产大幅贬值

---

## DEC-070

- 决策：确认“`150/171` 类 IoU 为 `0`”并不是因为训练子集缺类，而是因为 `COCO-Stuff-171` 的长尾极重；当前应将此视为诊断协议设计问题，而不是简单的实现错误或任务失败
- 结论：`keep`
- 原因：
  - 对当前 `1000` 图 anchor subset 统计后发现：
    - 实际覆盖了 `170 / 171` 个类
    - 只缺失 `waterdrops`
  - 因此“很多类根本没进训练集，所以 IoU=0”这个解释不成立
  - 但长尾极重：
    - `132` 个类只出现在 `<=50` 张图里
    - `150` 个类只出现在 `<=100` 张图里
    - 只有极少数常见 stuff / large-region 类拥有足够像素监督
  - 与 full-val 结果结合：
    - `aAcc=22.79` 明显高于 `mIoU=1.11`
    - 说明模型并非完全乱猜，而是在常见像素上已经学到一些东西
    - 只是 `mIoU` 对 `171` 个类等权平均，把大量零 IoU 长尾类全部算进去了
  - 修复后结果中：
    - `21` 个类 IoU 非零
    - 若只对这些非零类求均值，约为 `9.05`
    - 虽然仍不理想，但比表面上的 `1.11` 更接近真实训练状态
  - 因此当前最合理的研究动作是：
    1. 不再把 `COCO-Stuff-171 full mIoU` 直接当作第一阶段 supervised diagnostic 的唯一成败标准
    2. 降级 benchmark 到更低熵标签空间
    3. 同时允许内部诊断使用：
       - present-class / nonzero-class mIoU
       - frequent-class mIoU
       作为调试指标（不是最终论文主指标）

---

## DEC-071

- 决策：`VOC20` 可以作为当前 Phase 1 的降级 benchmark 候选，并且在“科学诊断清晰度”上很有吸引力；但若追求最小迁移成本，`COCO-Object` 仍是更自然的第一切换点
- 结论：`keep`
- 原因：
  - 本地仓库已现成支持 `VOC20`：
    - `segmentation/datasets/pascal_voc20.py`
    - `segmentation/configs/_base_/datasets/pascal_voc12_20.py`
    - 多个现有 config 已预留 `voc20` 数据集入口
  - `VOC20` 的优势：
    - 仅 `20` 类，熵显著低于 `COCO-Stuff-171`
    - 是 open-vocabulary segmentation 文献中的经典 benchmark
    - 很适合回答“feature-defined data composition 是否会影响 learner”这种诊断问题
  - `VOC20` 的代价：
    - 当前本地 `VOC2012` 数据目录尚未准备好
    - 更重要的是：当前 `.slicetune` 下已冻结的大量 feature / subset / manifest 资产是围绕 `COCO-Stuff` 构建的
    - 切到 `VOC20` 意味着需要重建：
      - feature extraction
      - subset manifests
      - intervention materialization
  - 相比之下，`COCO-Object`：
    - 仍在同一 COCO 数据源内
    - 当前仓库也已有 dataset 支持
    - 更能复用现有 COCO 系列资产
  - 因而当前建议分成两种模式：
    1. 若目标是“最快保留现有资产并降低难度”，优先 `COCO-Object`
    2. 若目标是“最干净、最简单的 segmentation 诊断 benchmark”，`VOC20` 完全可行，且值得做
  - 2026-04-15 本地复核：
    - 已确认 `segmentation/datasets/pascal_voc20.py` 与 `segmentation/configs/_base_/datasets/pascal_voc12_20.py` 可作为直接切换入口
    - 但 `data/VOCdevkit/VOC2012` 当前为空缺，若正式切换需先补齐数据与相应 feature/subset 资产

---

## DEC-072

- 决策：后续 Literature Radar 不再只围绕 `data slice / remix` 检索，而应并行覆盖四条与“根据数据特征提升数据质量、构造更适合当前场景的数据集”直接相关的文献线
- 结论：`keep`
- 四条主线：
  1. `training dynamics / data maps`
  2. `label quality / label error detection`
  3. `model-aware subset selection / data valuation / data curation`
  4. `distribution-shift / long-tail / multi-domain benchmark`
- 原因：
  - 当前项目最早强调的是 `slice-based model debug / model validation / data quality assessment`
  - 但新问题已经扩展到：是否能基于样本特征或模型导出信号，主动提升数据质量，并构造对当前场景更有利的数据集
  - 文献复核后发现，最相关且可为 SliceTune 提供直接启发的工作并不只来自 `slice` 方向，而主要来自以下代表方法：
    - `Dataset Cartography`：用训练动态刻画 easy / ambiguous / hard 区域
    - `Confident Learning` 与 segmentation 版 label-quality work：用模型置信度与错误估计做数据清洗
    - `GLISTER / GRAD-MATCH / CRAIG / Data Shapley / JEST / Beyond Neural Scaling Laws`：用 validation、梯度、数据价值或 reference model 做数据子集/数据质量优化
    - `WILDS / BREEDS / MESS / AUCSeg / DataPerf / DataComp`：提供更容易让“数据分布影响模型性能”显性化的 benchmark / task 设计灵感
  - 对当前项目的直接含义是：
    1. SliceTune 后续不应只依赖手工 secondary features
    2. 应考虑把 `interpretable features` 与 `model-derived signals` 结合
    3. 若当前 benchmark 对 feature effect 不敏感，应优先考虑 `long-tail / subpopulation shift / multi-domain` 更敏感的 setting，而不是立刻放弃 segmentation

---

## DEC-073

- 决策：可以考虑彻底切换 `image segmentation` 的 learner 与 dataset，但不应做“任意更换”；应优先切到“中等容量 supervised learner + 结构化分布敏感 benchmark”的组合，并用分阶段 2x2 设计保持可解释性
- 结论：`keep`
- 原因：
  - 当前 `CLIP-DINOiser` family 与 `COCO-Stuff-171` 的组合已经多次显示：
    - learner 先验过强或训练目标过间接时，数据组成效应不容易显现
    - 高熵 benchmark 会把任何早期 signal 淹没
  - 但这并不等于“必须放弃 segmentation”；更可能意味着：
    - 要换到一个对数据组成更敏感的 learner
    - 并同时换到一个更容易显现分布效应的 benchmark
  - 当前最有希望的成功区间不是：
    - 超强 foundation segmentation model + 简单 benchmark
  - 而是：
    - `moderate-capacity supervised learner`
    - `small/medium budget`
    - `long-tail / subpopulation shift / multi-domain / low-entropy segmentation benchmark`
  - 初步概率判断：
    1. 若只是换成更简单 benchmark（如 `VOC20`）但任务仍较同质，拿到“更稳定 baseline”的概率较高，但拿到“强 feature sensitivity”的概率只属中等
    2. 若切到“更结构化 shift”的 segmentation setting（如 long-tail / multi-domain），让当前 feature 产生可重复响应的概率更高
    3. 若直接换成超强 foundation model，反而可能再次把数据效应洗平，不利于 Phase 1 诊断
  - 因而推荐的切换原则是：
    1. 保留 `image segmentation`
    2. 切换到 `supervised learner`
    3. benchmark 优先选“低熵且有结构化 shift”的 setting
    4. 用 2x2 设计分开判断 learner effect 与 benchmark effect，而不是一次把所有变量混改

---

## DEC-074

- 决策：采纳“先寻找 `diagnostic benchmark–learner pair`”这一高层策略，但不直接照搬“`COCO-Object` 最快、`Cityscapes/ACDC` 最强 signal”作为立即执行计划；需先按本地数据准备度与代码接入成本做修正
- 结论：`keep`
- 采纳部分：
  - 将当前主证明从 `CLIP-DINOiser on COCO-Stuff-171` 暂时解绑
  - `CLIP-DINOiser` 保留为仓库血缘与回接验证，不再承担 Phase 1 主诊断 learner
  - Phase 1 主协议继续使用：
    - direct feature-guided intervention
    - `real / shuffled / matched-random` controls
    - local metrics 而不是只看 full global `mIoU`
  - `DeepLabV3+ R50-D8` 作为主 learner、`SegFormer-B0` 作为 challenger 的方向是合理的
- 需要修正的地方：
  - `COCO-Object` 在概念上确实是最低迁移成本路线，但此前我们对其前置条件判断过严
  - 现已确认：
    - 本地已有 `tools/convert_coco_object.py`
    - 它直接读取 `data/coco_stuff164k/annotations/{train2017,val2017}` 下现有 raw `*.png` masks
    - 过滤掉已生成的 `*TrainIds.png`
    - 通过 class-id remap 生成 `*_instanceTrainIds.png`
    - 不需要额外 COCO instance JSON 才能完成 object-only semantic label 生成
  - 当前 `research_harness/supervised_probe.py` 仍硬编码到 `COCO-Stuff164k` 与 `171` 类，尚未成为 benchmark-agnostic runner
  - `Cityscapes` 本地仅部分准备完成：
    - `gtFine` 已存在
    - `leftImg8bit` 仍未展开成可直接训练目录
    - `ACDC` 数据当前未准备
  - 因此“最快路线”和“最强 signal 路线”在研究判断上成立，但在工程上都还不是零成本直接运行
- 当前更稳的执行顺序：
  1. 先把 supervised probe 抽象成 dataset-aware runner
  2. 先运行 `tools/convert_coco_object.py` 生成 `COCO-Object` labels，并复用现有 COCO 系资产
  3. 若 `COCO-Object` 上 signal 仍弱，再推进 `Cityscapes/ACDC`
  4. `VOC20` 继续保留为最干净 sanity benchmark 候选

---

## DEC-075

- 决策：将当前 `DeepLabV3+ on COCO-Stuff-171` 结果正式标注为“diagnostic probe under borrowed architecture”，而不是“原论文 benchmark recipe 的 faithful reproduction”
- 结论：`keep`
- 证据：
  - 当前 supervised probe 入口 [research_harness/supervised_probe.py](/home/yuhe/clip_dinoiser/research_harness/supervised_probe.py:1) 直接借用的是 MMSeg 的 `deeplabv3plus_r50-d8_512x512_80k_ade20k.py` 作为起点
  - 当前 probe runner 硬编码到：
    - `COCOStuffDataset`
    - `171` 类
    - `1000` 图 manifest 子集训练
    - full COCO-Stuff validation
  - 本地 MMSeg 安装中未提供 `DeepLabV3+` 对 `COCO-Stuff164k` 的现成官方 config；现成的 `COCO-Stuff164k` configs 主要是 `DeepLabV3` / `PSPNet` / `BiSeNet`
  - 当前 `result.json`：
    - `mIoU=1.11`
    - `mAcc=2.64`
    - `aAcc=22.79`
    - `21` 个类别 IoU 非零
    - 非零类别平均 IoU 约 `9.05`
- 解释：
  - 这说明当前模型并非“完全随机”或“训练完全坏掉”
  - 更合理的解释是：
    1. 我们借用了 `DeepLabV3+` 架构
    2. 但放到了一个比原论文主 benchmark 更苛刻的本地诊断协议里：
       - `COCO-Stuff-171`
       - `1000` 图随机子集
       - 小预算训练
       - full-val 上按 `171` 类等权平均
    3. 在这个协议下，常见类与 stuff 大类已经开始被学习，但长尾类别几乎全部压成 `0`
  - 因此当前 `1.11 mIoU` 不能解释成：
    - `DeepLabV3+` 论文结果只有这个水平
    - 或 `DeepLabV3+` 本身不成立
  - 应解释成：
    - 当前 Phase 1 的 `benchmark–learner–metric` 契约对这个 probe 过于严厉
- 外部对照：
  - `DeepLabV3+` 原论文明确报告的是 `PASCAL VOC 2012` 与 `Cityscapes`，并在摘要中给出 test set performance `89.0%` 与 `82.1%`
  - 当前本地实验不是对这些原论文 benchmark recipes 的直接复现
- 进一步更正：
  - 此前我们还高估了 `COCO-Object` 的迁移门槛
  - 现已确认本地独立目录 [data/coco_object](/home/yuhe/clip_dinoiser/data/coco_object) 已存在并基本准备完成，不必再把 “先生成 `_instanceTrainIds.png`” 视作必须前置 gate

---

## DEC-076

- 决策：将 `VOC20` 视为最干净的 supervised sanity benchmark，将 `Cityscapes` 视为更有结构化 signal 潜力的 segmentation benchmark；同时明确 `CLIP-DINOiser` 与 `DeepLabV3+` 的论文数字不可直接按同一游戏规则比较
- 结论：`keep`
- 依据：
  - 本地 `VOC20` dataset 定义只有 `20` 个前景类，且 `reduce_zero_label=True`，背景被忽略：[pascal_voc20.py](/home/yuhe/clip_dinoiser/segmentation/datasets/pascal_voc20.py:9)
  - 本地 `COCO-Stuff164k` dataset 定义为 `171` 个语义类：[coco_stuff.py](/home/yuhe/clip_dinoiser/segmentation/datasets/coco_stuff.py:5)
  - `DeepLabV3+` 原论文摘要报告的是 `PASCAL VOC 2012` 与 `Cityscapes` test set 上的 `89.0%` 与 `82.1%`
  - 用户提供的 `CLIP-DINOiser` 论文表中，`CLIP-DINOiser` 行是 open-vocabulary / frozen CLIP family setting，而不是同等条件下的 fully supervised closed-set segmentation
- 解释：
  - `VOC20` 对当前 Phase 1 最有价值的是：
    - 类别熵低
    - object-centric
    - 更容易先验证“learner 是否能正常吃到 signal”
  - `Cityscapes` 虽然也比 `COCO-Stuff-171` 更低熵，但它真正吸引人的地方是：
    - 类别数低
    - 场景结构高度规则
    - 更容易把天气/能见度/夜间/小目标等分布效应显性化
  - `CLIP-DINOiser` 数字低于 `DeepLabV3+` 并不奇怪：
    1. 前者更接近 open-vocabulary / weakly supervised / frozen-backbone family
    2. 后者是 fully supervised benchmark-specific segmentation model
    3. 因而数值对比只能作为“上限感知”，不能直接作为算法优劣的同条件比较

---

## DEC-077

- 决策：开放词汇 segmentation 继续保留为更强最终主张候选，但不再承担 Phase 1 的第一性信号验证职责；Phase 1 先在更可解释的 supervised segmentation setting 中证明数据组成效应，再回接 open-vocabulary family
- 结论：`keep`
- 理由：
  - 若在开放词汇 / frozen-foundation family 上直接成功，论文主张会更强
  - 但当前它的问题是：
    1. learner 先验太强
    2. 训练路径更间接
    3. supervision 更弱
    4. 失败时极难判断到底是 feature 问题、benchmark 问题，还是 open-vocab learner 本身过于钝化
  - 因而对当前 Phase 1，开放词汇 segmentation 的主要缺点不是“不高级”，而是“解释性太差、失败歧义太大”
- 当前建议叙事：
  1. 先在 supervised diagnostic benchmark–learner pair 上证明：
     - `real intervention > shuffled/random`
     - 响应超过 noise floor
  2. 再把同一套 feature / intervention protocol 回接到 `CLIP-DINOiser` 或其他 open-vocabulary family
  3. 若回接成功，则论文叙事更强：
     - 不是只对简单闭集任务有效
     - 而是能迁移到更难的开放词汇 setting

---

## DEC-078

- 决策：将本地工作区重构为统一父目录 `~/slicetune`，并把官方 DeepLab 代码与 `clip_dinoiser` 并列放置
- 结论：`keep`
- 已执行：
  - 创建工作区根目录：`/home/yuhe/slicetune`
  - 将主仓库移动到：`/home/yuhe/slicetune/clip_dinoiser`
  - 为兼容旧绝对路径与历史 artifact，保留软链接：
    - `/home/yuhe/clip_dinoiser -> /home/yuhe/slicetune/clip_dinoiser`
  - 通过官方仓库 `https://github.com/tensorflow/models.git` 进行 sparse clone，并落到：
    - `/home/yuhe/slicetune/deeplab`
  - 当前 `deeplab` 工作树仅展开：
    - `research/deeplab`
- 解释：
  - 这样可以把 `clip_dinoiser` 与官方 `DeepLab` 代码放到同一工作区下，便于后续对照与迁移
  - 保留旧软链接可以避免当前 `.slicetune` 中大量历史绝对路径立即失效

---

## DEC-079

- 决策：将官方 `DeepLab` 论文结果复现流程明确拆分为“环境安装 / PYTHONPATH / dataset->TFRecord / 初始或发布 checkpoint / train / eval / vis / export”这一套 benchmark-specific 操作链，而不是把官方复现误解为单一 `train.py` 命令
- 结论：`keep`
- 依据：
  - `README.md` 明确把安装、PASCAL、Cityscapes、ADE20K、Model Zoo 分成独立文档入口
  - `model_zoo.md` 明确说明发布了用于 reproducing results 的 checkpoints 与 frozen graphs
  - `pascal.md` 与 `cityscapes.md` 分别给出 dataset conversion、train/eval/vis 的逐步命令
  - `local_test.sh` 给出了最小端到端示例：model_test -> dataset conversion -> init checkpoint -> train -> eval -> vis -> export
- 解释：
  - 后续若我们要“真正复现官方结果”，必须先选定 benchmark（如 PASCAL 或 Cityscapes）和对应 backbone / checkpoint recipe
  - 不能把当前在 MMSeg 中借用 `DeepLabV3+` 架构的 probe 结果误当成官方 TensorFlow repo 的直接复现

---

## DEC-080

- 决策：将官方 `DeepLab` 运行环境固定为 `tensorflow-gpu==1.15.5 + protobuf==3.20.3 + tf_slim`，并将 `research/slim` 视为官方 repo 运行所需的必备工作树，而不是可选组件
- 结论：`keep`
- 触发事实：
  - 在 `deeplab` conda 环境中直接运行 `python deeplab/model_test.py` 时，首先因 `protobuf==4.24.4` 报错：
    - `Descriptors cannot not be created directly`
  - 降级到 `protobuf==3.20.3` 后，第二个错误暴露为：
    - `ModuleNotFoundError: No module named 'nets'`
  - 检查发现当前官方 repo 仅 sparse checkout 了 `research/deeplab`，而 `local_test.sh` 与运行时导入实际还依赖 `research/slim`
  - 展开 `research/slim` 后，第三个错误暴露为：
    - `ModuleNotFoundError: No module named 'tf_slim'`
  - 安装 `tf_slim` 后，`deeplab/model_test.py` 成功通过：
    - `Ran 5 tests ... OK (skipped=1)`
- 依据：
  - `installation.md` 只泛化要求安装 `Tensorflow`，未 pin `protobuf`
  - `train.py` / `eval.py` 仍显式依赖 `tensorflow.contrib` 与 `tf.app.flags`，说明此代码路径属于 TF1 栈
  - `local_test.sh` 明确要求把 `research` 与 `research/slim` 同时加入 `PYTHONPATH`
- 解释：
  - 当前我们 clone 的是官方 `tensorflow/models` 的现代快照，而不是 2018 年冻结快照，因此需要显式补齐新环境下的兼容依赖
  - 后续任何“官方 DeepLab 可运行性”判断，都应以：
    - `TF1.15`
    - `protobuf 3.20.x`
    - `tf_slim`
    - `research/slim`
    作为最小前提

---

## DEC-081

- 决策：不再把 `local_test.sh` 中的 `data.deepai.org/PascalVOC2012.zip` 当成可靠数据入口；后续 `PASCAL VOC 2012` 复现优先采用“手工准备原始数据 + 手工执行转换脚本”的路径
- 结论：`keep`
- 触发事实：
  - `model_test.py` 通过后，运行 `local_test.sh` 卡在：
    - `https://data.deepai.org//PascalVOC2012.zip`
  - 当前服务器报错：
    - IPv4 连接超时
    - IPv6 不可达
- 依据：
  - `download_and_convert_voc2012.sh` 当前硬编码使用：
    - `BASE_URL="https://data.deepai.org/"`
    - `FILENAME="PascalVOC2012.zip"`
  - 该脚本是一个便捷下载脚本，不是 DeepLab 训练与评测逻辑本身
- 解释：
  - 这说明当前 `local_test.sh` 失败点在旧镜像地址可用性，而不是官方 DeepLab 代码链本身
  - 后续更稳的复现路径是：
    1. 手工准备 `VOC2012` 原始数据到期望目录
    2. 单独运行 `remove_gt_colormap.py`
    3. 单独运行 `build_voc2012_data.py`
    4. 再继续 `train.py / eval.py`

---

## DEC-082

- 决策：直接将官方 `download_and_convert_voc2012.sh` 的下载源切换到 PASCAL VOC 官网 `VOCtrainval_11-May-2012.tar`，并同步修正解压方式与目录根路径
- 结论：`keep`
- 已执行：
  - 将：
    - `BASE_URL="https://data.deepai.org/"`
    - `FILENAME="PascalVOC2012.zip"`
    替换为：
    - `BASE_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2012"`
    - `FILENAME="VOCtrainval_11-May-2012.tar"`
  - 将解压方式从：
    - `unzip`
    改为：
    - `tar -xf`
  - 将 `PASCAL_ROOT` 从：
    - `${WORK_DIR}/VOC2012`
    修正为：
    - `${WORK_DIR}/VOCdevkit/VOC2012`
- 解释：
  - 这不是改变数据契约，而是把旧镜像源替换成官方数据源
  - `build_voc2012_data.py` 真正要求的是：
    - `VOCdevkit/VOC2012/JPEGImages`
    - `VOCdevkit/VOC2012/SegmentationClass`
    - `VOCdevkit/VOC2012/ImageSets/Segmentation`
  - 因此使用官方 tar 反而比第三方 zip 更符合 DeepLab 原始数据结构

---

## DEC-083

- 决策：将 `PASCAL VOC 2012` 数据准备阶段标记为通过，后续官方 DeepLab 复现重点切换到“初始 checkpoint + 最小 train/eval”
- 结论：`keep`
- 已确认：
  - 已存在官方目录结构：
    - `VOCdevkit/VOC2012/JPEGImages`
    - `VOCdevkit/VOC2012/SegmentationClass`
    - `VOCdevkit/VOC2012/SegmentationClassRaw`
    - `VOCdevkit/VOC2012/ImageSets/Segmentation/{train,val,trainval}.txt`
  - 已生成 `TFRecord`：
    - `train-*`
    - `val-*`
    - `trainval-*`
- 当前统计：
  - `JPEGImages`: `17125`
  - `SegmentationClass`: `2913`
  - `SegmentationClassRaw`: `2913`
- 解释：
  - 这说明当前官方 DeepLab 复现不再卡在环境安装或数据准备
  - 下一步应优先验证：
    - 初始 checkpoint 下载是否可达
    - 最小 `train.py / eval.py` 是否可运行

---

## DEC-084

- 决策：将官方 `local_test.sh` 的 `PASCAL VOC` 最小链路视为已成功通过；后续不再把“官方 DeepLab 能否在本机正常工作”当作未决问题
- 结论：`keep`
- 触发事实：
  - `local_test.sh` 评测输出：
    - `eval/miou_1.0_overall[0.821973264]`
  - 与 `local_test.sh` 注释中的期望值基本一致：
    - `Using the provided checkpoint, one should expect mIOU=82.20%.`
- 解释：
  - 这说明：
    - 官方 checkpoint 下载可用
    - 官方 TensorFlow DeepLab 的最小 train/eval 链在当前服务器上可运行
    - 当前官方代码链已具备作为“可信 supervised learner 基线实现”的资格
  - 但这还不等于“完整从头训练复现论文主结果”：
    - 当前只是使用官方提供 checkpoint，并在 `local_test.sh` 约定下做 10-step 最小训练后评测

---

## DEC-085

- 决策：将 Phase 1 中优先关注的 supervised segmentation 数据特征收敛为 5 类主轴：
  1. 标注质量 / 边界噪声
  2. 类别长尾与像素不平衡
  3. 小目标比例 / 对象尺度
  4. 图像质量与 adverse conditions
  5. 域组成 / 分布偏移
- 结论：`keep`
- 依据：
  - 分割标签质量工作表明像素级标注错误会显著影响训练与评测，并且可以被模型分数有效识别
  - 长尾语义分割工作明确把类别不平衡视为核心问题
  - 小目标语义分割与相关 benchmark 分析反复指出 small-object regime 是稳定难点
  - ACDC / WILDS 一类工作表明天气、时间、地点等分布变化会显著拉低泛化表现
- 解释：
  - 后续若要为 SliceTune 选择更可能放大 signal 的 benchmark 与 feature family，应优先围绕这 5 类主轴建 probe

---

## DEC-086

- 决策：将“learner 是否对某条 feature 敏感”的判断标准固定为一个四件套，而不再单看全局 `mIoU`
- 结论：`keep`
- 当前 metric contract：
  1. `response_amplitude`
     - 定义：`Δ_real = metric(high) - metric(low)`
  2. `response_to_noise_ratio`
     - 定义：`RNR = |Δ_real| / sigma_noise`
     - 其中 `sigma_noise` 来自同 learner / 同 benchmark / 同预算 / 同 metric 的多 seed anchor 波动
  3. `control_gap`
     - 定义：`CG = |Δ_real| - max(|Δ_shuffled|, |Δ_random|)`
  4. `directional_consistency`
     - 定义：多个 seed 下 `sign(Δ_real)` 是否一致
- 当前 promote 规则：
  - 若同时满足：
    - `RNR > 1`
    - `CG > 0`
    - `directional_consistency >= 2/2` 或至少高一致
  - 则可初步判断当前 learner 对该 feature 有敏感性
- 当前补充说明：
  - `metric` 不应只用 global `mIoU`
  - 对局部 feature，还应同步看局部指标，例如：
    - `small-object mIoU`
    - `rare-class mIoU`
    - `present-class mIoU`
- 解释：
  - Phase 1 关心的是“特征干预是否产生超过噪声和随机对照的稳定响应”
  - 因而最关键的不是绝对分数高低，而是：
    - 响应幅度
    - 相对噪声强度
    - 相对 control 的优势

---

## DEC-087

- 决策：新 segmentation benchmark 的 feature extraction 不做“整套重写”，而采用 **通用 extractor 复用 + dataset adapter 重写 + 分层验证** 的迁移策略
- 结论：`keep`
- 依据：
  - 当前质量特征提取入口 [extract_quality_raw_features.py](/home/yuhe/slicetune/clip_dinoiser/extract_quality_raw_features.py:47) 实际只是薄包装，核心逻辑在 `QualityRawExtractor`，输入契约主要是 `subset_root + sample_index.npy`
  - 当前 coverage embedding 入口 [extract_coverage_embeddings.py](/home/yuhe/slicetune/clip_dinoiser/extract_coverage_embeddings.py:47) 本质上只依赖 `image_rel` 和图像路径，可跨 benchmark 复用
  - 当前干预物化器 [research_harness/feature_intervention.py](/home/yuhe/slicetune/clip_dinoiser/research_harness/feature_intervention.py:57) 已经把多维 processed records 合并后再按 axis score 物化 high/low 子集，说明上层协议并不强依赖 COCO-Stuff
  - 但 `small_object_ratio` 的现有实现 [difficulty.py](/home/yuhe/slicetune/clip_dinoiser/feature_utils/data_feature/implementations/difficulty.py:75) 默认写死了 COCO-Stuff 的 thing-id / ignore-index 先验；`class_presence` 读取 [class_coverage.py](/home/yuhe/slicetune/clip_dinoiser/slice_remix/class_coverage.py:10) 也默认 `_labelTrainIds.png` 协议
- 当前解释：
  - 因此切 benchmark 时不应推倒重写所有特征算法
  - 应优先：
    1. 重写 dataset adapter：
       - sample index builder
       - image / annotation path resolver
       - label id / ignore label / class count 协议
    2. 保留可 benchmark-agnostic 的 extractor：
       - sharpness / blur / quality 类
       - CLIP embedding / coverage embedding 类
    3. 对 label-derived features 做 dataset-aware 参数化：
       - class presence
       - rare-class coverage
       - small-object ratio
  - 当前验证要求：
    1. schema / range / empty-case 单元检查
    2. 手工视觉 spot-check
    3. 独立重算交叉验证
    4. 分布 sanity
    5. materialized high/low subset 的 face-validity 验证

---

## DEC-088

- 决策：以 `VOC20` 作为第一个新 benchmark adapter 落地模板，并优先补齐 “dataset spec + subset builder + class-presence remap”
- 结论：`keep`
- 当前已实现：
  1. `feature_utils/data_feature/dataset_specs.py`
     - 提供 `coco_stuff / coco_object / voc20 / cityscapes` 的轻依赖 dataset feature spec registry
  2. `tools/sample_voc20_subset.py`
     - 按 `ImageSets/Segmentation/*.txt` 构造 `sample_index.npy`
     - 不再错误地直接扫描 `JPEGImages`
  3. `slice_remix/class_coverage.py`
     - 支持 `annotation_rels`
     - 支持 `annotation_suffix`
     - 支持 `reduce_zero_label`
     - 因而可以正确处理 `VOC20` 的原始标签协议
  4. `run_feature_pipeline.py`
     - 支持通过 `--dataset-spec` 把 dataset metadata 注入 feature meta
- 当前验证：
  - 小范围定向测试 `10 passed`
  - 覆盖：
    - dataset spec contract
    - VOC20 subset sampling
    - class presence remap
    - CLI 参数解析
- 当前解释：
  - 这说明新 benchmark 迁移不需要先推倒整个 feature pipeline
  - 第一个可执行模板已经建立，后续 `Cityscapes` 可沿同一 adapter 思路推进

---

## DEC-089

- 决策：不直接粗暴搬走 `clip_dinoiser/feature_utils`，而是在 `~/slicetune` 根下先建立一个 **过渡期 root-level shared package**
- 结论：`keep`
- 当前实现：
  - 已创建：
    - `/home/yuhe/slicetune/feature/__init__.py`
    - `/home/yuhe/slicetune/feature/features/__init__.py`
    - `/home/yuhe/slicetune/feature/features/dataset_specs.py`
    - `/home/yuhe/slicetune/feature/README.md`
  - 当前公共包先桥接最轻依赖的公共接口：
    - `dataset_specs`
- 当前验证：
  - 在 `sys.path` 加入 `/home/yuhe/slicetune` 后，已成功 import：
    - `from feature.features import get_dataset_feature_spec`
  - 并成功读取：
    - `voc20.annotation_suffix == '.png'`
- 当前解释：
  - 这样做的目的不是“已经完成物理迁移”
  - 而是先把真正通用的公共入口稳定下来
  - 在保持 `clip_dinoiser` 现有实验链不被打断的前提下，逐步将 feature subsystem 升级为 workspace-level asset

---

## DEC-090

- 决策：将 `dataset_specs` 正式提升为 `~/slicetune/feature` 下的 root-level source-of-truth，并让 `clip_dinoiser` 本地模块退化为兼容桥接
- 结论：`keep`
- 当前实现：
  - root-level 实现：
    - `/home/yuhe/slicetune/feature/features/dataset_specs.py`
  - repo-level bridge：
    - [dataset_specs.py](/home/yuhe/slicetune/clip_dinoiser/feature_utils/data_feature/dataset_specs.py:1)
- 当前验证：
  - `pytest` 小回归通过：
    - `11 passed`
  - 已验证：
    - `from feature.features import get_dataset_feature_spec`
    - `from clip_dinoiser.feature_utils.data_feature.dataset_specs import get_dataset_feature_spec`
    - 两条导入链返回一致 spec
- 当前解释：
  - 这意味着 workspace-level shared feature package 已经不只是空壳
  - 它已经开始承载第一批真正独立的公共实现
  - 后续可按同样模式迁移：
    - `bundle`
    - `postprocess`
    - `pipeline`

---

## DEC-091

- 决策：将 workspace 根目录共享包名称从 `slicetune_features` 统一重命名为 `feature`
- 结论：`keep`
- 当前实现：
  - 目录已重命名：
    - `/home/yuhe/slicetune/feature`
  - repo-level 兼容桥接 import 已对齐：
    - [dataset_specs.py](/home/yuhe/slicetune/clip_dinoiser/feature_utils/data_feature/dataset_specs.py:1)
  - root/repo 桥接测试文件已同步改名：
    - [test_root_feature_bridge.py](/home/yuhe/slicetune/clip_dinoiser/tests/test_root_feature_bridge.py:1)
- 当前解释：
  - 这样命名更短，也更符合后续把 feature extraction 逐步提升为 workspace-level 公共基础设施的目标
  - 当前仍保留“root-level source-of-truth + repo-level compatibility bridge”的迁移策略，不改变公共边界设计

---

## DEC-092

- 决策：先将 `~/slicetune/deeplab` 官方 TensorFlow DeepLab 视为 **可信外部基线与 recipe reference**
- 结论：`keep`
- 当前事实：
  - `model_test.py` 已通过
  - 官方 `VOC2012` 下载/转换链路已跑通
  - 最小评测结果已达到官方 `local_test` 预期：
    - `eval/miou_1.0_overall = 0.821973264`
- 当前解释：
  - 这足以证明：
    - 官方 DeepLab 代码可在当前服务器环境稳定运行
    - 该方法可作为可信 supervised segmentation reference
  - 但 TensorFlow1 + TFRecord + benchmark-specific 文档式流程不适合作为当前 Phase 1 的高频实验底座
  - 因此最初判断是：
    - 官方 repo 保持为 reference / benchmark sanity lane
    - `clip_dinoiser` 内的 supervised probe runner 继续承担主实验 lane
    - 优先推进 dataset-aware benchmark adapter 与 feature intervention protocol

---

## DEC-093

- 决策：根据当前用户明确偏好，将 supervised segmentation 主实验 lane 正式切到官方 `~/slicetune/deeplab` 框架
- 结论：`keep`
- 当前原因：
  - 用户已明确指出，既然官方 DeepLab 已经跑通，就不应继续优先调自实现 probe runner
  - 官方 DeepLab 代码已经在本机完成最小可信复现，具备作为主训练框架的资格
  - 将 feature extraction 抬升到 workspace-level shared package，本来也是为了更方便接入官方 DeepLab
- 当前执行含义：
  - 后续优先研究：
    - 如何在官方 benchmark-specific 数据准备层接入 feature-aware subset / intervention
    - 如何复用官方 `train.py / eval.py` 跑 `real / control` 对照
  - `clip_dinoiser` 内 probe runner 降级为备用实现，不再承担第一优先级主证明职责

---

## DEC-094

- 决策：在官方 DeepLab 的 `VOC` lane 中，优先采用 `train_aug` 作为训练池，`val` 作为固定评测集；Phase 1 先做固定预算的 feature-aware 子集对照，而不是直接全量 `train_aug` 重训
- 结论：`keep`
- 当前原因：
  - `train_aug` 更接近官方强 recipe，也比 `train` 更适合作为 SliceTune 的可干预训练池
  - 直接全量 `train_aug` 重训成本更高，而且更容易把早期 feature signal 平均掉
  - 固定 `val` 不参与训练，能保持本地评测口径稳定
- 当前执行含义：
  - 优先构造：
    - `anchor`
    - `feature_high`
    - `feature_low`
    - `matched_random`
  - 所有条件保持：
    - 相同样本数
    - 相同官方 DeepLab 训练 recipe
    - 相同 `val` 评测集
  - 技术接入点优先放在：
    - split 文本生成
    - 对应 `TFRecord` 物化
    - `--train_split=<custom_split>`

---

## DEC-095

- 决策：官方 DeepLab `VOC` lane 的 `train_aug` 不自行从 SBD 原始数据重建，第一轮先采用 canonical `train_aug.txt + SegmentationClassAug` 资产落盘
- 结论：`keep`
- 当前原因：
  - 当前目标是尽快把 feature-aware data selection 接入官方 DeepLab，而不是复刻一套 SBD overlap 清洗流程
  - canonical `train_aug` 资产更接近社区常用 `VOC2012_aug / train_aug` 契约
  - 直接使用现成 split 清单与增强标注，能显著降低 `val` 泄漏、mask 协议不一致和路径组织错误的风险
- 当前事实：
  - `train_aug.txt` 已下载到官方 `ImageSets/Segmentation/`
  - `SegmentationClassAug/` 已解压到官方 `VOC2012/`
  - `train_aug.txt` 行数为 `10582`
  - 当前缺失 mask 数为 `0`
- 当前下一步：
  - 将 `SegmentationClassAug` 转为 `SegmentationClassAugRaw`
  - 生成 `train_aug-*.tfrecord`
  - 再在完整 `train_aug` 上做 `2000` 图的 `anchor/high/low/random` feature-aware split

---

## DEC-096

- 决策：官方 DeepLab `VOC train_aug` lane 已通过数据准备 gate，当前正式进入 `2000` 图固定预算的 feature-aware split 生成阶段
- 结论：`keep`
- 当前事实：
  - `SegmentationClassAugRaw` 已生成完成
  - `train / val / trainval / train_aug` 四套 TFRecord 均已生成，各 `4` 个 shard
  - 当前不再需要继续处理 VOC augmentation 数据准备
- 当前执行含义：
  - 现在主变量应切换到训练子集组成：
    - `anchor_2000`
    - 后续 `feature_high_2000`
    - `feature_low_2000`
    - `matched_random_2000`
  - 官方 DeepLab 继续只作为训练 / 评测底座，不再改数据转换协议

---

## DEC-097

- 决策：对官方 DeepLab `VOC` lane 采用“最小 custom split 兼容补丁”，而不是重命名覆盖官方内置 split
- 结论：`keep`
- 当前原因：
  - 用户希望继续使用官方 `train.py / eval.py` 作为主执行底座
  - 当前 SliceTune 需要 `anchor/high/low/random` 这类自定义 split 名
  - 官方 `data_generator.py` 默认只白名单：
    - `train`
    - `train_aug`
    - `trainval`
    - `val`
  - 直接覆写官方 split 名会污染 benchmark 契约，且不利于并行保留基线
- 当前实现：
  - 若 `split_name` 不在白名单中，但 `dataset_dir` 下存在匹配的 `<split>-*.tfrecord`，则允许继续训练
- 当前效果：
  - `anchor_2000_seed0` 这类自定义 split 可以直接通过：
    - `--train_split=anchor_2000_seed0`

---

## DEC-098

- 决策：将“官方 DeepLab 已能跑通”与“官方 DeepLab 已能在 GPU 上高效训练”明确区分；在继续正式实验前，必须单独通过 `TF1 GPU runtime` gate
- 结论：`keep`
- 当前原因：
  - 当前 `deeplab` 环境中的 `tensorflow-gpu==1.15.5` 虽能导入并跑 `VOC local_test`，但实机检查显示：
    - `tf.test.is_gpu_available()` 为 `False`
    - TensorFlow 反复报缺 `libcudart.so.10.0 / libcublas.so.10.0 / libcudnn.so.7` 等老运行时库
    - 日志明确输出 `Cannot dlopen some GPU libraries` 与 `Skipping registering GPU devices...`
  - 这说明当前训练慢的主要原因是：
    - `TF1.15` 没有真正注册 GPU
    - 训练主体仍主要在 CPU 上运行
- 当前执行含义：
  - 当前若继续在该状态下跑 `anchor/high/low/random` 正式矩阵，实验成本会被不可接受地放大
  - 因此在继续官方 DeepLab 主实验前，应先尝试最小修复：
    - 为 `deeplab` 环境补齐 `cudatoolkit=10.0.130 + cudnn=7.6.5`
    - 然后重新验证 GPU 可用性
  - 若补齐后仍无法稳定注册 GPU，则应重新评估：
    - 官方 TF1 DeepLab 是否只保留为 recipe/reference lane
    - 而不是继续承担高频实验底座职责

---

## DEC-099

- 决策：确认官方 `deeplab` 环境已通过 `TF1 GPU runtime` gate；后续官方 DeepLab `VOC` lane 可继续作为当前 supervised 主实验底座
- 结论：`keep`
- 当前原因：
  - 在 `deeplab` conda 环境中补齐：
    - `cudatoolkit=10.0.130`
    - `cudnn=7.6.5`
  - 并令：
    - `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:...`
    后，TensorFlow 1.15 已成功注册四张 4090
  - 当前验证结果：
    - `tf.test.is_gpu_available() == True`
    - `device_lib.list_local_devices()` 已出现 `/device:GPU:0..3`
    - 缺失 `libcudart.so.10.0 / libcudnn.so.7` 的问题已消失
- 当前执行含义：
  - 后续官方 DeepLab `anchor/high/low/random` 训练不再需要先怀疑 GPU runtime
  - 当前第一优先级重新回到：
    - 跑通 `anchor_2000_seed0` 的正式单卡 baseline
    - 量化其速度与 `val` 指标
  - 同时将环境约束固定为：
    - 每次训练前必须先 `conda activate deeplab`
    - 每次训练前必须先 `export LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}\"`

---

## DEC-100

- 决策：将“GPU 设备已注册成功”与“官方 TF1 DeepLab 可稳定训练”分开判断；当前官方 GPU 训练 lane 暂不视为已通过
- 结论：`limited keep`
- 当前原因：
  - 在 GPU runtime 修复后，TensorFlow 1.15 已能成功注册 `/device:GPU:0..3`
  - 但当前两条训练 smoke 都出现异常：
    - 自定义 `anchor_2000_seed0`：首步前长时间停在 `global_step/sec: 0`，并出现 `Loss is inf or nan`
    - 官方 `trainval`：也出现同类现象
  - 已排除：
    - 标签值域脏数据
    - custom split 白名单问题
    - 缺失老 CUDA/cuDNN runtime
- 当前解释：
  - 当前最合理判断已从“数据协议问题”转向：
    - `TF1.15 + CUDA10.0/cuDNN7 + RTX4090` 的训练期兼容性/数值稳定性问题
  - 因此不能仅因为 GPU 已注册，就继续把官方 GPU 训练视为可直接扩张的主实验 lane
- 当前执行含义：
  - 先做 CPU 对照 smoke 以完成因果收敛
  - 若 CPU 正常、GPU 异常，则应将：
    - 官方 DeepLab 保留为 reference / recipe / eval lane
    - 而不是高频 feature-aware 训练底座

---

## DEC-101

- 决策：将官方 TensorFlow DeepLab 的训练异常正式收敛为 **cuDNN 卷积算法损坏导致的 GPU 状态污染**，不再继续按“数据协议可能有错”处理
- 结论：`keep diagnosis / park GPU training lane`
- 当前原因：
  - 真正的 CPU-only 1-step debug 训练已成功完成，且：
    - 输入统计正常
    - 标签统计正常
    - `logits/pixel_loss/loss` 全部有限
  - 对应的 GPU debug 训练中，TensorFlow 明确报出：
    - `Detected cudnn out-of-bounds write in convolution buffer!`
    - `This is likely a cudnn bug`
  - 随后下一步的 debug loss 统计显示：
    - `logit_min = 3.40282347e+38`
    - `logit_max = -3.40282347e+38`
    - `total_loss = nan`
    - `loss = nan`
  - 即当前 NaN 的直接前因不是数据本身，而是 cuDNN 卷积路径已经把 GPU state 写坏
- 当前执行含义：
  - 官方 DeepLab 的 GPU 训练 lane 不再作为 SliceTune 的默认主实验底座
  - 当前可保留：
    - 官方 DeepLab 代码库作为 recipe / eval / reference lane
    - 已加入的 debug 开关作为后续问题复现与证据保留
  - 当前不应继续默认投入：
    - `anchor/high/low/random` 的大规模官方 GPU 训练扩张

---

## DEC-102

- 决策：将 supervised segmentation 主执行底座切回 modern PyTorch / MMSeg runner，并把该 runner 正式扩成 dataset-aware
- 结论：`keep`
- 当前原因：
  - 官方 TensorFlow DeepLab 的 GPU 训练 lane 已确认会触发 cuDNN 卷积 buffer 越界写
  - 继续将其作为高频主实验底座会污染：
    - baseline 可信度
    - feature intervention 结果解释
    - 实验推进速度
  - 仓库中已有 mmseg-based supervised probe 雏形，迁移成本明显低于继续硬顶 TF1
- 当前执行含义：
  - `research_harness/supervised_probe.py` 现改为支持：
    - `coco_stuff`
    - `voc20`
    - `cityscapes`
  - 现支持：
    - full-train baseline
    - manifest-defined subset 训练
  - 当前新的推荐顺序：
    1. 先跑 modern PyTorch `voc20` full baseline
    2. 再跑 `cityscapes` full baseline
    3. baseline 站住后再接 feature-aware subset 实验

---

## DEC-103

- 决策：将 `VOC` lane 的 full baseline 默认向 `train_aug` 对齐，并新增标准 `voc`（21 类）入口以避免把 `VOC20` 与官方 VOC 指标直接混比
- 结论：`keep`
- 当前原因：
  - 首轮 `voc20` baseline 实际只使用了 `train.txt` 的 `1464` 张，并只跑了 `2000` iter、`batch=2`
  - 这与 mmseg / 官方常见 DeepLab VOC recipe 的主要条件有大幅偏差：
    - `train_aug=10582`
    - `20k` 级别 schedule
    - 更大的有效 batch
  - 若不先把训练池与 benchmark 口径对齐，会把“训练预算 mismatch”误判成“实现 bug”
- 当前执行含义：
  - `voc20` 现会优先使用：
    - `train_aug.txt`
    - `SegmentationClassAug`
  - 同时新增：
    - `--dataset voc`
  - 后续若要逼近官方/标准 VOC 结果，优先从 `voc` baseline 开始，而不是把 `voc20` 直接对标官方 VOC test 数字

---

## DEC-104

- 决策：将 `voc train_aug` 上得到的 modern PyTorch `73.51 mIoU / 2.34h` baseline 视为当前 supervised segmentation lane 的可执行锚点，并把后续 feature 实验预算默认下调到 screening 级别
- 结论：`keep`
- 当前原因：
  - `voc` full baseline 已证明：
    - 训练协议正确
    - 模型可稳定收敛
    - 分数已远高于“学不起来”区间
  - 但 wall-clock 约 `2.34h` / run
  - 若直接把所有 `anchor/high/low/random × seeds` 都跑满 `20k`，实验吞吐会过低
- 当前执行含义：
  - `20k` 保留为：
    - baseline / confirmatory run budget
  - feature sensitivity 第一轮默认改为：
    - `5k` 或 `10k` screening budget
  - 当前主问题从“训练栈是否可用”转为：
    - 哪个最小预算仍能保留 feature response signal

---

## DEC-105

- 决策：将 modern PyTorch supervised probe runner 正式扩成支持多卡分布式训练的主执行入口
- 结论：`keep`
- 当前原因：
  - `voc train_aug` 单卡 `20k` baseline 已证明训练协议可用，但 wall-clock 约 `2.34h/run`
  - 当前瓶颈已从“训练是否正确”转为“feature intervention 吞吐是否足够”
  - 现有 runner 已基于 mmseg，接入 mmcv distributed API 的工程风险明显低于继续自造多卡训练循环
- 当前执行含义：
  - `research_harness/supervised_probe.py` 现支持：
    - `init_dist(...)`
    - distributed `train_segmentor`
    - distributed `multi_gpu_test`
    - 仅 rank0 写出最终 `result.json`
  - `run_supervised_probe_experiment.py` 现支持：
    - `--launcher`
    - `--dist-backend`
    - `--gpu-collect`
    - `--local-rank/--local_rank`
  - 当前推荐的多卡启动方式为：
    - `torchrun --nproc_per_node=<N> run_supervised_probe_experiment.py ... --launcher pytorch`

---

## DEC-106

- 决策：distributed supervised probe 路径中保留 `SyncBN`，只在单卡路径中降级为普通 `BN`
- 结论：`keep`
- 当前原因：
  - 单卡 probe 中将 `SyncBN -> BN` 是合理简化
  - 但在多卡 DDP 路径中继续强制替换，会让实际 recipe 偏离标准 mmseg distributed 设定
  - 首轮 `2` 卡 `200 iter` smoke 的低精度不应和这种 recipe 偏移混淆
- 当前执行含义：
  - `build_supervised_probe_cfg(..., preserve_syncbn=True)` 已用于 distributed 路径
  - 单卡 baseline 的现有行为保持不变
  - 后续多卡对齐实验应基于：
    - distributed + `SyncBN`
    - 再评估 wall-clock 与早期收敛形态

---

## DEC-107

- 决策：将 `2` 卡 DDP `200 iter` smoke 的当前解释冻结为“工程可行性通过”，而不是“精度协议已完成验证”
- 结论：`keep`
- 当前原因：
  - 最新结果：
    - `mIoU=9.94`
    - `train_seconds=63.687`
    - `total_seconds=89.307`
  - 与先前 `2` 卡 `200 iter` 普通 `BN` smoke 相比，保留 `SyncBN` 后结果形态基本一致
  - 这说明在仅 `200 iter` 的极短预算下：
    - 该 smoke 足以验证 DDP 路径可用
    - 但不足以作为 recipe 质量或 DDP 精度收益的强证据
- 当前执行含义：
  - 后续若要判断多卡 recipe 的真实质量与 speedup，应优先跑：
    - 单卡 vs 2卡 的同预算对照
    - 推荐 `1000 iter`
  - 当前 `200 iter` smoke 主要用于：
    - 验证 runner / rank0 eval / result writeout

---

## DEC-108

- 决策：在 `voc train_aug` full baseline 站住后，下一阶段主线切换为“先对完整 `train_aug` 提 feature，再做受控 subset 对照实验”
- 结论：`keep`
- 当前原因：
  - 当前 full baseline 已回答“训练链能否正常工作”
  - 当前真正未回答的问题是：
    - feature 是否对应可重复的训练响应
  - 若继续只在 full `train_aug` 上堆更多 baseline，并不能直接提供更强的 feature-sensitive 证据
- 当前执行含义：
  - feature 必须先在完整 `train_aug` 候选池上提取，而不是在事先抽出的子集上提
  - 第一轮对照默认采用：
    - `anchor`
    - `high`
    - `low`
    - `matched_random`
  - 第一批 feature 轴默认采用：
    - `small_object_ratio`
    - `rare_class_coverage`
  - 第一轮预算默认采用：
    - pilot=`1000 iter`
    - confirmatory=`5000 iter`

---

## DEC-109

- 决策：将 `2` 卡 DDP + `SyncBN` + `1000 iter` 冻结为当前 `voc train_aug` feature screening 的默认执行协议
- 结论：`keep`
- 当前原因：
  - 单卡 `1000 iter` 结果：
    - `mIoU=53.13`
    - `train_seconds=427.59`
    - `total_seconds=462.597`
  - `2` 卡 DDP + `SyncBN` `1000 iter` 结果：
    - `mIoU=52.10`
    - `train_seconds=283.651`
    - `total_seconds=308.855`
  - 当前 `ΔmIoU≈1.03`，但 wall-clock 提升约 `1.5x`
  - 该差距足以支持当前阶段的 pilot screening，而不必继续停留在单卡吞吐上
- 当前执行含义：
  - 后续第一轮 feature 对照实验默认采用：
    - `2` 卡 DDP
    - `SyncBN`
    - `1000 iter`
    - 总 batch=`8`
  - 只有在某 feature 轴显示出明确响应后，才升到更高预算确认

---

## DEC-110

- 决策：将 `voc train_aug` 的 full-pool feature extraction 与 subset manifest materialization 收敛为独立包层服务，而不是继续堆顶层临时脚本
- 结论：`keep`
- 当前原因：
  - 当前主线已经从“修训练底座”切到“准备 feature-sensitive 对照实验”
  - 下一步的核心不是再写一批 ad-hoc 命令，而是稳定产出：
    - `feature_table`
    - `summary`
    - `anchor/high/low/matched_random` manifests
  - 该逻辑既属于数据契约，也会被后续多轮实验反复复用
- 当前执行含义：
  - 已新增：
    - `slice_remix/voc_feature_subsets.py`
    - `tools/prepare_voc_train_aug_feature_experiment.py`
  - 当前默认首批轴为：
    - `small_object_ratio`
    - `rare_class_coverage`
  - 后续若要扩展更多 VOC `train_aug` feature 轴，应优先继续补进这一包层服务，而不是新增同类散乱脚本

---

## DEC-111

- 决策：将当前 `small_object_ratio` pilot 的结论冻结为“response exists but axis interpretation not yet promotable”
- 结论：`park`
- 当前原因：
  - `anchor_2000`：`mIoU=52.24`
  - `high_2000`：`mIoU=35.71`
  - `low_2000`：`mIoU=33.72`
  - `matched_random_2000`：`mIoU=54.75`
  - 当前 `high/low` 都显著低于 `anchor`，而 `matched_random` 反而略优于 `anchor`
  - 这说明：
    - learner 对 subset 组成有强响应
    - 但该响应目前不是 clean monotonic feature response
  - 当前补充诊断也显示：
    - `high/low` 伴随明显类存在模式与每图前景类数量漂移
    - 因而当前结果更像 extreme materialization / composition drift 效应，而不是单纯 `small_object_ratio` 轴效应
- 当前执行含义：
  - `small_object_ratio` 不应直接 promote 为当前主 feature claim
  - 但也不应立刻 kill：
    - 它仍然提供了“training protocol is composition-sensitive”的正证据
  - 下一步应优先：
    - 跑 `rare_class_coverage` pilot
    - 若还要回头验证 `small_object_ratio`，先收紧 control / exclusivity / matching 约束

---

## DEC-112

- 决策：将当前项目的主研究对象显式收敛为“非单调、多特征耦合、human-in-the-loop 的 feature distribution insight / steering”
- 结论：`keep`
- 当前原因：
  - 用户已明确澄清：
    - 项目不是要证明某条特征单调越高越好或越低越好
    - 真实目标是理解：
      - 多种 feature 如何共同作用
      - 为什么过高或过低都不理想
      - 什么样的混合分布更适合当前任务
  - 当前 `small_object_ratio` pilot 也与这一目标一致：
    - 单轴极端推进并未带来更优结果
    - 更好的状态更像来自中间或混合分布
- 当前执行含义：
  - 后续实验不应只按“单轴单调可优化”标准裁决
  - 应显式纳入：
    - 非单调最优区间
    - 多特征耦合效应
    - 用户偏好驱动的 distribution steering
  - 系统层最终更应服务于：
    - 帮助用户理解当前训练任务更适合偏向哪些数据特征
    - 而不是承诺找到全局最优分布

---

## DEC-113

- 决策：将 `matched_random` 收紧为和同轴 `anchor/high/low` 互斥，并补充轻量结果汇总工具
- 结论：`keep`
- 当前原因：
  - 先前 VOC pilot 中，`matched_random` 与同轴 `high/low` 存在数百样本重叠
  - 这不会导致实验失效，但会削弱 control 的解释力度
  - 当前项目既然以用户 insight 和 distribution steering 为目标，就应优先保持 control 的可解释性
- 当前执行含义：
  - `prepare_voc_train_aug_feature_experiment(...)` 生成的新 manifests 中：
    - `matched_random` 将与同轴 `high/low` 和 `anchor` 保持互斥
  - 已新增：
    - `tools/summarize_supervised_probe_results.py`
  - 后续进入 `rare_class_coverage` 或重跑 `small_object_ratio` 前，应先重新生成一版 feature-prep artifacts

---

## DEC-114

- 决策：将当前 `rare_class_coverage` strict pilot 的结论冻结为“balanced anchor dominates extremes; supports non-monotonic mixture hypothesis”
- 结论：`keep`
- 当前原因：
  - `anchor_2000`：`mIoU=53.62`
  - `high_2000`：`mIoU=40.53`
  - `low_2000`：`mIoU=14.58`
  - `matched_random_2000`：`mIoU=41.34`
  - 当前 `anchor` 明显优于三种偏置/对照子集
  - 该结果不支持：
    - “把 rare class coverage 拉高就更好”
    - 或“拉低就更好”
  - 但它支持：
    - 当前任务下更优状态更像 balanced mixture
    - 单轴极端推进会显著伤害训练
- 当前执行含义：
  - `rare_class_coverage` 不应被 promote 为单独主优化轴
  - 但这轮结果可以正向支持项目主张：
    - 非单调
    - 多特征耦合
    - 需要围绕 anchor 附近做更细粒度 steering
  - 下一步应优先从：
    - extreme high/low
    - 转向中等幅度 perturbation 与两轴局部联合实验

---

## DEC-115

- 决策：新增一份仅含客观事实的最近实验汇总文档，作为后续写作与对齐的引用底稿
- 结论：`keep`
- 当前原因：
  - 用户要求将最近实验、训练协议、评测协议、数据集、任务场景以及既有特征指标与语义整理为单独 markdown
  - 当前已将相关事实汇总到：
    - `.slicetune/logs/2026-04-19_recent_experiment_fact_sheet.md`
- 当前执行含义：
  - 后续若需要引用最近 VOC supervised probe 实验的客观事实，应优先引用该文档

---

## DEC-116

- 决策：将下一阶段 feature validation 主线收敛为“三阶段筛选”，而不是单轴线性全扫到底
- 结论：`keep`
- 当前原因：
  - 现有 `small_object_ratio` 与 `rare_class_coverage` pilot 都已显示：
    - 训练对数据组成敏感
    - 极端 `high/low` 不足以支撑单轴单调解释
  - 当前项目目标也已澄清为：
    - 非单调
    - 多特征耦合
    - preference-aware distribution steering
- 当前执行含义：
  - 下一阶段默认采用：
    1. coarse screening
    2. strict validation
    3. local interaction / shape experiment
  - 当前候选特征池建议冻结在 `10–12`
  - 当前主系统目标特征数建议冻结在约 `6`
  - 当前工程上应优先实现：
    - generic feature registry
    - generic VOC feature table extraction
    - generic subset materialization for multiple feature axes

---

## DEC-117

- 决策：将 VOC feature-prep 的单文件实现拆成包层模块，并把旧入口保留为兼容桥接
- 结论：`keep`
- 当前原因：
  - 原 `slice_remix/voc_feature_subsets.py` 同时承载：
    - 数据契约
    - feature 计算
    - subset 选择
    - manifest 写出
    - summary 写出
  - 这会继续鼓励把新逻辑堆回单文件
- 当前执行含义：
  - 当前实现已拆分为：
    - `contracts.py`
    - `dataset.py`
    - `scoring.py`
    - `service.py`
    - `__init__.py`
  - `slice_remix/voc_feature_subsets.py` 现只保留 backward-compatible re-export
  - `tools/prepare_voc_train_aug_feature_experiment.py` 现支持：
    - `--feature-axis`
  - 后续新增 VOC feature axis 时，默认应补进：
    - `slice_remix/voc_feature_prep/scoring.py`
    - 而不是回到 bridge 或 CLI 中堆实现

---

## DEC-118

- 决策：优先接入一批 mask-native segmentation feature axes，先扩充 screening 候选池，再考虑 image-quality 族的统一接入
- 结论：`keep`
- 当前原因：
  - 当前 segmentation 主线最容易直接对齐的数据是 mask 与 label 协议
  - mask-native 特征：
    - 更贴近当前监督分割任务
    - 不依赖额外图像预处理或新依赖
    - 更适合作为 feature screening 基础层
- 当前执行含义：
  - 当前已新增：
    - `foreground_class_count`
    - `pixel_class_entropy`
    - `foreground_area_ratio`
    - `foreground_component_count`
    - `component_fragmentation`
  - 当前默认 screening 轴仍保持：
    - `small_object_ratio`
    - `rare_class_coverage`
  - quality 族如：
    - `laplacian`
    - `bga`
    - `noise_pca`
    将作为下一批接入，而不是和本次包层重构混在同一修改面
