# SliceTune Memory Index

本文件是长时 agent 会话的短索引文件。

目标：

- 作为新的长时 session 的快速入口
- 只保留高价值、低噪声、稳定的重要事实
- 把详细内容下沉到 `program / playbook / board / decision_log / logs`

## 当前锚点

- 仓库：`clip_dinoiser`
- 主 case：`image segmentation`
- 当前 phase：`Feature Signal Audit / Learner Sensitivity Audit`
- 当前主 benchmark：`COCO-Stuff`
- 当前主 learner：`CLIP-DINOiser`

## 当前最重要事实

- `EXP-P1-001` 已完成 noise floor 汇总
- 当前 global floor：`count=192`、`mean=24.2939`、`std=0.0260`、`range=0.14`
- `EXP-P1-002` 已完成固定 subset 多 training seed 审计
- 当前 training noise：`count=5`、`mean=24.2860`、`std=0.0089`、`range=0.0200`
- `EXP-P1-002` 已被人类放行为 `completed`
- `EXP-P1-003` 已完成 learner sensitivity ladder
- 当前 learner ladder 结果：
  - `fast_cached_1ep=24.29`
  - `fast_1ep=20.39`
  - `standard_3ep=20.75`
  - `regime_range=3.90`
- `EXP-P1-004` 已自动 materialize，并已冻结为最小 `learner adaptability audit`
- `EXP-P1-004` 当前固定：
  - learner variants=`L0_head_only / L1_task_head_plus / L2_last_block_partial`
  - probe axes=`quality_sharpness / difficulty_small_object`
  - control families=`real / shuffled / matched-random`
  - reporting metrics=`composition_response_amplitude / response_to_noise_ratio / directional_consistency / feature_validity_advantage`
- `EXP-P1-004` 的 `Tier A` executable runtime 已接通：
  - selective unfreeze 已进入训练脚本
  - `feature_intervention_matrix` 已成为 executable loop kind
  - 当前由 daemon 在 `clipdino2` runtime 下执行
- 当前 `EXP-P1-004` 的已知运行状态：
  - debate gate 已修复并通过
  - preflight false-negative 已修复（`clipdino2` 现可正确识别 CUDA）
  - 当前处于长前处理 / 早期执行阶段，尚未产出最终 `progress.json / cell_results.json`
- 当前判断：training noise 明显低于 global floor，可继续推进 learner sensitivity / feature intervention 审计
- 当前 harness 已具备：
  - task-level state machine
  - auto debate
  - auto propose
  - context packet
  - runtime preflight / heartbeat / reclaim / retry guard
  - human review stop + release
  - watchdog-ready queue runner
  - agentic planning / analysis / judgment artifact
  - executable literature radar with OpenAlex retrieval
  - design-mode-aware agentic planner for `EXP-P1-004`

## 必读主文件

1. `AGENTS.md`
2. `.slicetune/context/program.md`
3. `.slicetune/context/playbook.md`
4. `.slicetune/state/board.md`
5. `.slicetune/state/decision_log.md`

## 当前下一步

1. 先将 `feature_intervention_matrix` 接成 `Tier A` executable loop
2. 让该 loop 直接消费现有 `design_pack / evaluation_rubric / design_spec`
3. 在 runtime 中先支持：
   - learner-specific noise floor
   - 两个 probe axes
   - `real_feature_guided`
4. 持续观察 `EXP-P1-004` 的 early-runtime 产物是否开始落盘（cache / manifests / runs）
