# experiments 目录说明

本目录用于存放：

- 已注册实验卡
- 实验索引
- 实验结果索引
- 与实验相关的结构化摘要

当前版本先保留为空目录骨架。
后续建议按如下方式扩展：

- `cards/`
- `runs/`
- `summaries/`
- `judgements/`
`experiments/` 用于存放机器可读的 experiment card、实验索引，以及后续可由 `research_harness` 直接消费的最小注册对象。

当前约定：

- `*.json`：机器可读 experiment card
- `README.md`：目录说明

常用字段补充：

- `priority`：调度优先级，数值越小越优先
- `depends_on`：依赖的 experiment id 列表
- `attempt_count`：已执行 attempt 数
- `max_attempts`：允许的最大 attempt 数，`0` 表示不限制
- `last_attempt_id`：最近一次 attempt id
- `status=planned`：表示 proposer 已 materialize，但尚未进入可执行队列

第一版目标不是把所有实验都迁入这里，而是先让需要自动执行的关键 loop 具备稳定入口。

controller 约定：

- `run_research_queue.py` 可直接扫描本目录中的 `status=queued` 实验卡
- 若实验卡带有 `requires_debate=true`，则默认必须通过 debate gate
- 若实验卡带有 `human_review_required=true` 或 `phase_completion_candidate=true`，则成功后默认自动停在人工审核点
