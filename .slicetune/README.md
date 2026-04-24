# .slicetune 说明

`.slicetune/` 是本项目的本地研究记忆与运行骨架目录。

它的作用不是存放训练产物本身，而是存放：

- 当前研究上下文
- 当前工程规范
- 当前 phase 约束
- 当前任务与实验队列
- 决策日志
- 阶段日志
- 跨会话 handoff
- 标准模板

目录说明：

- `MEMORY.md`：长时 agent 会话的短索引
- `context/`：少量主文件，承担研究计划与总执行手册
- `state/`：当前执行面与决策面
- `logs/`：阶段观察与运行日志
- `experiments/`：实验注册与实验索引
- `judge_policies/`：独立 judge policy，避免 proposer 直接调阈值
- `runtime/`：机器可读 controller policy
- `approvals/`：机器可读人类审核开关
- `debates/`：debate 产物
- `handoffs/`：跨会话交接
- `templates/`：实验卡、judge 报告等模板

当前版本为中文审阅版，后续可统一替换为英文正式版。

建议阅读顺序：

1. `AGENTS.md`
2. `MEMORY.md`
3. `context/program.md`
4. `context/playbook.md`
5. `state/board.md`
6. `state/decision_log.md`
