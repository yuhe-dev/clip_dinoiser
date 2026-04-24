# Debate Decision: SCHEDULER-V1

## 决定

`approve`

## 采纳内容

- priority + depends_on + max_attempts + attempt_count
- scheduler readiness / queue snapshot
- attempt manifest 与 artifact snapshot
- daemon loop
- retry limit block

## 暂缓内容

- 完整的 attempt 级独立 canonical output root
- 自动生成新 experiment card 的 proposer agent
- 更复杂的 budget-aware multi-worker scheduling

## 通过理由

- 这是从“单次可执行 harness”迈向“长时间自治研究 runtime”的最小必要层。
- 该设计延续了代码 gate 优先于 prompt gate 的原则。
- 它能和当前 Phase 1 实验链共存，不需要推倒现有 loop。
