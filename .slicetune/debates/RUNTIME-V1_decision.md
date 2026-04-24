# Debate Decision: RUNTIME-V1

## 决定

`approve`

## 采纳内容

- 引入 runtime profile registry
- controller 预检 worker runtime
- 失败异常原子收口到 `failed_execution`
- 多 seed 复用改为 completion sentinel
- lease 补充 `pid / hostname / runtime_profile_id`

## 暂缓内容

- attempt 级目录的全面重构
- 更复杂的跨主机 lease 身份校验

## 通过理由

- 该设计直接解决了 `EXP-P1-002` 的真实阻塞。
- 该设计与用户要求的“代码约束优先于 prompt 约束”一致。
- 该设计为后续长时自治研究循环补齐了最关键的一层可靠性基础。
