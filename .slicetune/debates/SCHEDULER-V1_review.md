# Review Card: SCHEDULER-V1

## Reviewer 关注点

1. scheduler 必须避免下游抢跑上游，不能只按文件名排序。
2. stale reclaim 不能误回收仍在运行的进程。
3. retry 必须有上限或 dead-letter 语义，否则 daemon 容易无限试错。
4. attempt 必须保留独立证据，不能把不同尝试的结果完全混在一起。
5. human review stop 必须是硬停止，不是日志提醒。

## 反对的偷懒方案

- 继续用文件名顺序决定谁先跑。
- 只给卡片记状态，不给 attempt 留证据。
- 失败后无限手动/自动回到 `queued`。
- 在没有 dependency 的情况下默认让 daemon 随便推进下游卡。

## 最小验收标准

1. scheduler 至少支持 `priority + depends_on`。
2. attempt 至少有独立 manifest 和 artifact snapshot。
3. retry 达到上限时，卡片进入明确 blocked 状态。
4. daemon 必须在 human review stop 触发时退出。
