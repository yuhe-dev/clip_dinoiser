# Review Card: RUNTIME-V1

## Reviewer 关注点

1. 失败必须原子收口，不能再出现卡片停在 `running` 但 worker 已失败的状态污染。
2. preflight 应在拿 lease 前执行，避免把已知坏环境也记成正式运行。
3. resume 不能只凭 `result.json` 存在，需要成功 sentinel。
4. stale reclaim 需要至少记录 `pid / hostname / runtime_profile_id`，避免误回收。

## 反对的偷懒方案

- 继续把 `python` 当成环境选择方式。
- 只修 prompt，不修状态机与代码 gate。
- 只在成功分支写 checkpoint。
- 只凭输出文件存在就把 seed 判为完成。

## 最小验收标准

1. worker 异常时，controller 必须把卡片落到明确终态。
2. runtime profile 选择和 preflight 结果必须有落盘证据。
3. 多 seed 任务必须能区分“成功完成的旧结果”和“半截残留结果”。
4. 成功晋升后仍保留 human review stop，不允许 controller 越权推进 phase。
