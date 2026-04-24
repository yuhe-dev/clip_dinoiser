# 最新 Handoff

日期：2026-04-15

## 本轮完成内容

- 确认 `EXP-P1-004` 的科学运行实际上已经全部完成，但 card 曾因 judge 参数兼容问题停在 `failed_execution`
- 修复 `judge_feature_intervention_matrix` 与 policy kwargs 的兼容性
- 将 `EXP-P1-004` 重新入队并完成最终 finalize
- 当前正式产物已全部落盘：
  - [result_bundle.json](/home/yuhe/clip_dinoiser/artifacts/research_harness/EXP-P1-004_feature_intervention_matrix/result_bundle.json)
  - [judge_report.json](/home/yuhe/clip_dinoiser/artifacts/research_harness/EXP-P1-004_feature_intervention_matrix/judge_report.json)
  - [analysis_brief.json](/home/yuhe/clip_dinoiser/artifacts/research_harness/EXP-P1-004_feature_intervention_matrix/agentic/analysis_brief.json)
  - [judgment_brief.json](/home/yuhe/clip_dinoiser/artifacts/research_harness/EXP-P1-004_feature_intervention_matrix/agentic/judgment_brief.json)
- 新增一条真正接入 CLIP backbone 梯度的实验链路：
  - 新配置：[feature_experiment_fast_cached_slide_backbone_grad.yaml](/home/yuhe/clip_dinoiser/configs/feature_experiment_fast_cached_slide_backbone_grad.yaml)
  - 已修改 `get_clip_features / MaskClip.extract_feat` 支持 `track_grad`
  - 已修复 `extract_v` 的原地加法反向传播问题
  - 已用 smoke test 确认 `decode_head.proj / last block / ln_post` 均收到非零梯度

## 当前最重要状态

1. `EXP-P1-001` 已完成：global random-subset noise floor 很窄
2. `EXP-P1-002` 已完成：fixed-subset training noise 更窄
3. `EXP-P1-003` 已完成：更适合解释为 `protocol sensitivity audit`
4. `EXP-P1-004` 已完成：正式结论为 `park`
5. `EXP-P1-004` 的关键科学发现：
   - `L0/L1/L2` 三个 learner variants 的 noise floor 一致，均为 `24.29`
   - 两条 probe axes 的 response 在三个 learner 上也一致：
     - `quality_sharpness` amplitude=`0.15`
     - `difficulty_small_object` amplitude=`0.13`
   - 因此当前轻量 selective-unfreeze ladder 没有拉开 learner 对数据组成的敏感性差异
6. 当前更适合的解释是：
   - `Tier A` screen 已完成
   - 当前 ladder 未区分 learners
   - 还不能宣称 learner adaptability 已建立
7. 当前已经具备新的 stronger learner audit 入口，不必再沿旧的 `feats_idx=-3 + detached hook` 继续解释 `L1/L2`
8. 第二次手工 backbone-grad run 已确认新增 CLIP 模块真正收到梯度，但该具体 `L2 + final-feature` 组合结果为 `mIoU=23.22`，暂未显示正收益
9. 已新增一个手工命令拼装工具：
   - [tools/run_backbone_grad_manual.py](/home/yuhe/clip_dinoiser/tools/run_backbone_grad_manual.py)
   - 用于快速生成同一新配置下 `L0/L1/L2` 与不同 manifests 的可复制命令
10. 已新增一条独立的 `intermediate-grad learner family`
   - 配置：[feature_experiment_fast_cached_slide_intermediate_grad.yaml](/home/yuhe/clip_dinoiser/configs/feature_experiment_fast_cached_slide_intermediate_grad.yaml)
   - family 定义：
     - `L0`: `obj_proj + bkg_decoder`
     - `L1`: `L0 + resblocks.-3`
     - `L2`: `L1 + resblocks.-4`
   - 已完成随机输入单步反传验证：
     - `resblocks.-3: 8/12 params_with_grad`
     - `resblocks.-4: 12/12 params_with_grad`
   - 说明这条路已技术可跑，不再是“名义上解冻”

## 若下一会话接手，优先做什么

1. 阅读：
   - `.slicetune/state/board.md`
   - `.slicetune/state/decision_log.md`
   - `artifacts/research_harness/EXP-P1-004_feature_intervention_matrix/result_bundle.json`
   - `artifacts/research_harness/EXP-P1-004_feature_intervention_matrix/judge_report.json`
   - `artifacts/research_harness/EXP-P1-004_feature_intervention_matrix/cell_results.json`
2. 先做研究决策，而不是继续盲跑：
   - 路线 A：推进 `Tier B` controls，验证 real feature 是否优于 shuffled/random
   - 路线 B：扩 learner 分支，寻找比当前 `L0/L1/L2` 更有区分度的 adaptation mechanism
   - 当前更现实的下一步是：在同一 backbone-grad 新配置下补 `L0` 与 `L1`，形成可比较的 `L0/L1/L2`
   - 然后再在同一新配置下跑 `quality_sharpness / difficulty_small_object` 的 high/low 对照
3. 在没有新证据前，不要把 `EXP-P1-004` 解释成“现有 learner adaptability ladder 已经成功”
4. 如果继续推进 stronger learner audit，当前最优先的不是再扩 `final-feature` family，而是先跑 `intermediate-grad` family 的 4 卡并行 anchor：
   - `L0 anchor seed0`
   - `L1 anchor seed0`
   - `L2 anchor seed0`
   - `L2 anchor seed1`
5. 跑完后优先读取：
   - `artifacts/manual_runs/intermediate_grad_L0_anchor_seed0/result.json`
   - `artifacts/manual_runs/intermediate_grad_L1_anchor_seed0/result.json`
   - `artifacts/manual_runs/intermediate_grad_L2_anchor_seed0/result.json`
   - `artifacts/manual_runs/intermediate_grad_L2_anchor_seed1/result.json`
