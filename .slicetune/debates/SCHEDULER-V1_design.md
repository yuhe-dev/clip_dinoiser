# Design Card: SCHEDULER-V1

## 主题

把当前 queue runner 从“按文件顺序找 queued 卡”升级为“带优先级、依赖、attempt 记录和 daemon loop 的自动科研调度器”。

## 背景

- 当前系统已有 `phase / debate / runtime / human review` gate。
- 但如果没有 scheduler、attempt 记录和 daemon loop，系统仍然更像单次实验执行器，而不是长时间自治研究循环。
- 用户明确要求 agent system 能持续运行数小时，并且在需要人类决策时自动停下。

## 提议

1. Experiment card 新增：
   - `priority`
   - `depends_on`
   - `attempt_count`
   - `max_attempts`
   - `last_attempt_id`
2. 新增 `research_harness/scheduler.py`：
   - readiness 判定
   - dependency-aware 选卡
   - queue snapshot
3. 新增 `research_harness/attempts.py`：
   - attempt id
   - attempt manifest
   - attempt artifact snapshot
4. 新增 `run_research_daemon.py`：
   - 连续扫描卡片
   - 在 `continue / wait / stop` 三态之间切换
5. queue runner 在正式执行前记录 attempt，在结束后写回 terminal summary。

## 预期收益

- 系统不再只按文件名顺序选卡。
- 每次执行都有独立 attempt 记录和证据快照。
- daemon 可以持续轮询和推进多张实验卡。
- retry 有上限，不会无限重复消耗 GPU。

## 风险

- 当前 canonical output 仍以 card-level output_dir 为主，attempt 级隔离还不是完全重构版。
- 若依赖关系没有显式写进卡片，scheduler 仍无法替代研究设计本身。
- daemon 若配置过于激进，仍可能把太多卡片快速推进到 blocked 状态。

## 最小落地范围

- `research_harness/contracts.py`
- `research_harness/scheduler.py`
- `research_harness/attempts.py`
- `run_research_queue.py`
- `run_research_daemon.py`
