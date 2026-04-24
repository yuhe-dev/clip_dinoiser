# AGENTS.md

## 项目身份

本仓库当前被作为 **SliceTune** 的后端研究锚点来推进，重点不是通用前端系统，也不是只做一个模型 benchmark，而是逐步收敛出一个可 defend 的 **基于数据 slice 的训练分布优化与人机协同研究对象**。

当前默认锚点：

- 仓库：`clip_dinoiser`
- 主 case：`image segmentation`
- 主数据与验证环境：`COCO-Stuff` 相关训练与评测流程

长期目标：

- 产出一个可以投稿到 HCI / AI 顶会的、证据充分的算法与系统论文对象

短期目标：

- 先证明当前定义的 feature / slice / remix 对真实训练结果是否具有可重复的响应信号

---

## 总体推进原则

### 1. 按 pipeline 顺序推进

默认研究链条：

1. feature extraction
2. feature embedding / clustering / slice construction
3. human preference-aware slice weight remix candidate generation
4. materialization / practical validation
5. surrogate / refinement / iteration

下游阶段不能替代上游阶段的不足。

### 2. 证据优先于猜测

不要仅凭“看起来合理”冻结后续结构。
任何阶段的主要设计都应尽量通过实验、对照、稳定性审计或文献证据收敛。

### 3. 分支必须收敛到 keep / park / kill

每个主要研究分支都应明确趋向下列之一：

- `keep`：当前值得持续投入
- `park`：暂时保留，但不作为当前主线
- `kill`：当前证据不足或方向错误，应停止扩张

### 4. 优先做能产生证据的工作

优先级高于：

- 前端 polish
- 叙事包装
- surrogate 细调
- 大规模自动化本身

### 5. 当前主文件优先

如果本文件与下列文件冲突，以这些主文件为当前执行准则：

- `.slicetune/context/program.md`
- `.slicetune/context/playbook.md`

---

## 当前默认研究重心

当前默认重心不是整条 pipeline 全面推进，而是优先确认：

- 当前训练协议是否对数据组成变化敏感
- 当前 secondary features 是否真的影响训练结果
- 当前 slice 是否是一个有干预杠杆的优化对象

在这些问题没有形成明确证据前，下游模块默认处于 **影子模式** 或 **暂停扩张模式**。

---

## 新会话必读顺序

每次新的有效工作会话开始时，默认按如下顺序读取：

1. `AGENTS.md`
2. `.slicetune/MEMORY.md`
3. `.slicetune/context/program.md`
4. `.slicetune/context/playbook.md`
5. `.slicetune/state/board.md`
6. `.slicetune/state/decision_log.md`

然后只读取与当前 phase 直接相关的最小代码与 artifact。

---

## `.slicetune/` 记忆系统

`.slicetune/` 是本项目的本地研究记忆层，用于把会话外的科研状态显式持久化。

预期结构：

- `.slicetune/context/`：半稳定的上下文与契约
- `.slicetune/state/`：当前任务、队列、决策、分支状态
- `.slicetune/logs/`：阶段日志、实验观察摘要
- `.slicetune/experiments/`：实验注册与实验索引
- `.slicetune/handoffs/`：跨会话 handoff
- `.slicetune/templates/`：规范模板

不要将密码、token、私密凭证写入 `.slicetune/`。

---

## 会话输出要求

每次有效工作会话应至少产出下列之一：

- 代码或脚本修改
- 实验计划更新
- 实验结果结构化摘要
- 决策日志更新
- 分支状态调整
- 明确的下一步建议

不要以只有模糊文字总结结束会话。

---

## 持久化更新要求

每次有效会话结束后，至少更新：

- `.slicetune/state/board.md`
- `.slicetune/state/decision_log.md`
- 当前 phase 对应日志

若当前 phase 发生变化，还必须更新：

- `.slicetune/context/program.md`

若项目对象发生明显变化，还必须更新：

- `.slicetune/context/program.md`

---

## Agent 协作纪律

本项目采用 **阶段 owner 制 + 独立 judge 制 + 状态显式化** 的运行方式。

高层原则：

- 一次高成本实验只允许一个主要修改面
- Judge 不直接改算法代码
- State Keeper 不解释算法优劣，只负责持久化事实
- Literature Radar 只负责调研与接入建议，不直接覆盖主线
- 下游 agent 在上游 gate 未通过前不得抢跑主线

具体角色分工见：

- `.slicetune/context/playbook.md`

---

## 编辑纪律

- 保持改动与当前 phase 相关
- 不同时大改算法假设与评测协议
- 不隐式引入新的主指标或新的研究主张
- 若发现当前规范与仓库现实不一致，优先更新 `.slicetune/` 并记录理由

---

## 工程纪律

本项目后续不再默认接受“为了赶验证先堆很多顶层小脚本”的做法。

默认工程方向：

- 业务逻辑进入包模块
- CLI 文件保持薄包装
- 数据契约显式化
- I/O 与核心计算分离
- 优先高内聚、低耦合
- 优先可测、可替换、可复用

必须遵守的最小规则：

1. 顶层脚本不应承载大段核心逻辑
2. 一个模块应尽量只有一个主要变更原因
3. 不将数据加载、算法计算、artifact 写出、绘图逻辑混写在同一个大函数中
4. 新功能优先补进已有包层，而不是继续增加同类散乱脚本
5. 重构应分阶段进行，避免一次性推倒

详细规则见：

- `.slicetune/context/playbook.md`
- `.slicetune/state/board.md`

---

## 当前状态说明

本文件是第一版中文审阅稿。
在结构与职责边界确认后，可以统一替换为英文正式版。
