# proposals 目录说明

本目录用于存放自动 proposer 生成的结构化提案。

当前约定：

- `*.json`：单条 proposal record
- proposal 默认来自 `.slicetune/runtime/proposal_policy.json`
- `draft_only` proposal 可自动 materialize 为 `planned` experiment card，但不会自动进入 `queued`

设计原则：

- proposer 只能沿着当前 phase 已锁定的分支和模板生成提案
- proposer 不能自由发明新的主张、新阶段或新 research branch
- 重要 proposal 仍需 debate / human review 才能转为正式运行
