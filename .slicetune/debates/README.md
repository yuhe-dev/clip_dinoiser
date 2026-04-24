`debates/` 用于存放机器可读或可审阅的 debate 产物。

推荐最小组成：

- design card
- review card
- debate decision

如果某个 experiment card 标记了 `requires_debate=true`，则默认还需要一个机器可读 debate bundle，并满足：

- `decision = approve`
- `round_count >= controller_policy.min_debate_rounds`

这样 queue/controller 才会放行。
