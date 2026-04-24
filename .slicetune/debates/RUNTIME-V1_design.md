# Design Card: RUNTIME-V1

## 主题

将 worker 运行时从裸 `python` 字符串升级为可注册、可预检、可审计的 runtime profile。

## 背景

- `EXP-P1-002` 首次真实运行因 base env 缺失 `torchvision / mmcv` 失败。
- 原 controller 在 `run_research_tick` 抛异常时没有把卡片原子收口到失败终态。
- 当前系统若要支持多小时自治，必须先把环境选择、依赖预检、失败恢复做成代码约束。

## 提议

1. 新增 `.slicetune/runtime/runtime_profiles.json` 作为 worker runtime 注册表。
2. controller 在执行前解析 `runtime_profile_candidates`，做只读 preflight。
3. preflight 通过后才允许卡片进入 `claimed / running`。
4. `run_research_tick` 与 `multi_seed` 使用解析后的 `python_bin`，不再依赖 shell PATH。
5. 多 seed 复用改为 `result.json + completion.json` 双条件，避免假 resume。

## 预期收益

- 明确锁定 `clipdino2` 作为当前 segmentation worker runtime。
- 运行失败更可诊断，状态更不容易污染。
- 后续长时自治时，controller 能先筛掉环境不合格的任务。

## 风险

- profile registry 若长期不维护，会引入新的漂移。
- preflight 过严可能阻挡某些只做分析、不需要 GPU 的任务。
- 当前 attempt-level 目录还未完全引入，仍需后续加强。

## 最小落地范围

- `research_harness/runtime_profiles.py`
- `research_harness/preflight.py`
- `run_research_queue.py`
- `run_research_tick.py`
- `research_harness/multi_seed.py`
