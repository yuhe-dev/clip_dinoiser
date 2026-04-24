# EXP-P1-002 审查卡

## 主要反对意见

1. 只做一个 subset 可能代表性不足。
2. fast config 的训练噪声可能低估或高估正式协议。
3. 如果不复用已有 deterministic manifest，实验定义会漂移。

## 结论

- `approve`

## 必须修改项

- 必须复用已有稳定 subset manifest
- 必须将 seed 列表、judge policy、run manifest 全部固定为机器可读对象
- 成功后若结果足以影响 phase 判断，必须停在人工审核点
