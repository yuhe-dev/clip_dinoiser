# EXP-P1-002 设计卡

## 问题

当前已经知道不同 `1000` 图随机子集的 global `mIoU` 动态范围很窄，但还不知道这部分波动有多少来自训练噪声。

## 提案

固定一个已有 subset manifest，不改变数据组成，只改变 `training seed`，重复训练多次，估计当前 learner / training protocol 的训练噪声。

## 为什么现在做

如果 training noise 已经接近跨 subset 波动，那么后续 feature / slice / recommendation 的小幅变化就必须非常谨慎解释。

## 风险

- 单个 subset 不一定代表全体 subset
- fast config 的噪声结构不一定等价于更长训练
- 实验成本明显高于 `EXP-P1-001`
