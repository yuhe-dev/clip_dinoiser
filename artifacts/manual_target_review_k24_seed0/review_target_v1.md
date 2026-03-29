# 显式 Target Portrait 版 Prior Graph / Beam Search 审查

## 运行配置
- atlas: `/home/yuhe/clip_dinoiser/artifacts/slice_canonical_vmf_auto_k24`
- projected_dir: `/home/yuhe/clip_dinoiser/artifacts/report_prep_gmm_k8/projected`
- portrait_source: `semantic`
- baseline_seed: `0`
- budget: `1000`
- target_mode: `raw_pool`
- target block: `quality.laplacian`
- target shift mass: `0.080`
- baseline gap to target: `0.377749`

## Target 编辑摘要
- target source: `pool_initialized`
- shape target bins: `[0.03718584403395653, 0.039142780005931854, 0.03999049961566925, 0.04690400883555412, 0.06055554375052452, 0.07993677258491516, 0.10507645457983017, 0.13207827508449554, 0.15256087481975555, 0.14687037467956543, 0.1041707843542099, 0.05552779138088226]`

## Prior Graph Top Edges
- `slice_05 -> slice_22` | score=0.3520, fit=0.3998, bias=0.0000, risk=0.1910, side=0.0000, boundary=0.5557, support_empty=0.0172, band=(0.024, 0.024)
- `slice_18 -> slice_04` | score=0.3464, fit=0.4037, bias=0.0000, risk=0.2290, side=0.0000, boundary=0.6153, support_empty=0.0717, band=(0.019, 0.019)
- `slice_18 -> slice_17` | score=0.3412, fit=0.3962, bias=0.0000, risk=0.2200, side=0.0000, boundary=0.6153, support_empty=0.0448, band=(0.019, 0.019)
- `slice_18 -> slice_22` | score=0.3366, fit=0.3906, bias=0.0000, risk=0.2159, side=0.0000, boundary=0.6153, support_empty=0.0325, band=(0.019, 0.019)
- `slice_14 -> slice_04` | score=0.3349, fit=0.3896, bias=0.0000, risk=0.2191, side=0.0000, boundary=0.5848, support_empty=0.0724, band=(0.021, 0.021)
- `slice_18 -> slice_11` | score=0.3292, fit=0.3840, bias=0.0000, risk=0.2194, side=0.0000, boundary=0.6153, support_empty=0.0427, band=(0.019, 0.019)
- `slice_18 -> slice_14` | score=0.3202, fit=0.3765, bias=0.0000, risk=0.2252, side=0.0000, boundary=0.6153, support_empty=0.0602, band=(0.019, 0.019)
- `slice_18 -> slice_00` | score=0.3192, fit=0.3742, bias=0.0000, risk=0.2199, side=0.0000, boundary=0.6153, support_empty=0.0444, band=(0.019, 0.019)
- `slice_18 -> slice_06` | score=0.3093, fit=0.3651, bias=0.0000, risk=0.2233, side=0.0000, boundary=0.6153, support_empty=0.0545, band=(0.019, 0.019)
- `slice_18 -> slice_16` | score=0.2993, fit=0.3564, bias=0.0000, risk=0.2286, side=0.0000, boundary=0.6153, support_empty=0.0706, band=(0.019, 0.019)
- `slice_18 -> slice_19` | score=0.2947, fit=0.3517, bias=0.0000, risk=0.2278, side=0.0000, boundary=0.6153, support_empty=0.0681, band=(0.019, 0.019)
- `slice_02 -> slice_04` | score=0.2923, fit=0.3496, bias=0.0000, risk=0.2295, side=0.0000, boundary=0.6122, support_empty=0.0762, band=(0.019, 0.019)

## Risk 审查
### 风险最高的边
- `slice_22 -> slice_15` | score=0.0292, fit=0.1114, bias=0.0000, risk=0.3290, side=0.0000, boundary=0.9408, support_empty=0.0463, band=(0.002, 0.002)
- `slice_22 -> slice_21` | score=-0.3061, fit=-0.2239, bias=0.0000, risk=0.3287, side=0.0000, boundary=0.9408, support_empty=0.0453, band=(0.002, 0.002)
- `slice_22 -> slice_04` | score=-0.2679, fit=-0.1858, bias=0.0000, risk=0.3285, side=0.0000, boundary=0.9408, support_empty=0.0446, band=(0.002, 0.002)
- `slice_22 -> slice_03` | score=-0.3383, fit=-0.2562, bias=0.0000, risk=0.3283, side=0.0000, boundary=0.9408, support_empty=0.0440, band=(0.002, 0.002)
- `slice_22 -> slice_16` | score=-0.1974, fit=-0.1154, bias=0.0000, risk=0.3281, side=0.0000, boundary=0.9408, support_empty=0.0434, band=(0.002, 0.002)
- `slice_22 -> slice_19` | score=-0.3578, fit=-0.2760, bias=0.0000, risk=0.3273, side=0.0000, boundary=0.9408, support_empty=0.0410, band=(0.002, 0.002)

### 风险最低的边
- `slice_01 -> slice_20` | score=0.0279, fit=0.0697, bias=0.0000, risk=0.1671, side=0.0000, boundary=0.4802, support_empty=0.0210, band=(0.030, 0.032)
- `slice_01 -> slice_22` | score=0.1060, fit=0.1478, bias=0.0000, risk=0.1674, side=0.0000, boundary=0.4802, support_empty=0.0220, band=(0.030, 0.032)
- `slice_01 -> slice_13` | score=0.0883, fit=0.1303, bias=0.0000, risk=0.1681, side=0.0000, boundary=0.4802, support_empty=0.0241, band=(0.030, 0.032)
- `slice_01 -> slice_11` | score=-0.0170, fit=0.0257, bias=0.0000, risk=0.1708, side=0.0000, boundary=0.4802, support_empty=0.0322, band=(0.030, 0.032)
- `slice_01 -> slice_09` | score=-0.0047, fit=0.0381, bias=0.0000, risk=0.1711, side=0.0000, boundary=0.4802, support_empty=0.0330, band=(0.030, 0.032)
- `slice_01 -> slice_00` | score=0.0523, fit=0.0951, bias=0.0000, risk=0.1713, side=0.0000, boundary=0.4802, support_empty=0.0339, band=(0.030, 0.032)

## Beam Search 层级统计
- depth 0: beam_in=1, expanded=46, deduped=46, beam_out=10, stopped=None, pruned={"repeat_edge": 0, "infeasible": 23, "proposal_pruned": 390, "no_gain": 2}
  评语：该层展开偏宽，dedup 后仍保留了较多状态，后续可继续收紧 proposal。
- depth 1: beam_in=10, expanded=513, deduped=513, beam_out=10, stopped=None, pruned={"repeat_edge": 10, "infeasible": 231, "proposal_pruned": 3890, "no_gain": 28}
  评语：该层展开偏宽，dedup 后仍保留了较多状态，后续可继续收紧 proposal。
- depth 2: beam_in=10, expanded=531, deduped=523, beam_out=10, stopped=None, pruned={"repeat_edge": 10, "infeasible": 267, "proposal_pruned": 3890, "no_gain": 154}
  评语：该层展开偏宽，dedup 后仍保留了较多状态，后续可继续收紧 proposal。
- depth 3: beam_in=10, expanded=159, deduped=155, beam_out=10, stopped=None, pruned={"repeat_edge": 10, "infeasible": 415, "proposal_pruned": 3890, "no_gain": 88}
  评语：该层展开偏宽，dedup 后仍保留了较多状态，后续可继续收紧 proposal。

## Candidate 审查
- 候选 1: progress=0.9192, opportunity=0.0402, complexity=0.1145, priority=0.8268, support=4, plan=[{"donor": 19, "receiver": 12, "amplitude": 0.010575637221336365, "score": 0.16570396599658294}, {"donor": 21, "receiver": 15, "amplitude": 0.010575637221336365, "score": 0.19385910677392895}]
- 候选 2: progress=0.9400, opportunity=0.0093, complexity=0.1407, priority=0.8246, support=6, plan=[{"donor": 1, "receiver": 11, "amplitude": 0.010575637221336365, "score": -0.01698116651985247}, {"donor": 3, "receiver": 12, "amplitude": 0.010575637221336365, "score": 0.1309794998330114}, {"donor": 21, "receiver": 15, "amplitude": 0.010575637221336365, "score": 0.19385910677392895}]
- 候选 3: progress=0.9153, opportunity=0.0387, complexity=0.1151, priority=0.8228, support=4, plan=[{"donor": 3, "receiver": 12, "amplitude": 0.010575637221336365, "score": 0.1309794998330114}, {"donor": 21, "receiver": 15, "amplitude": 0.010575637221336365, "score": 0.19385910677392895}]
- 候选 4: progress=0.9414, opportunity=0.0098, complexity=0.1455, priority=0.8223, support=6, plan=[{"donor": 3, "receiver": 12, "amplitude": 0.010575637221336365, "score": 0.1309794998330114}, {"donor": 17, "receiver": 15, "amplitude": 0.010575637221336365, "score": 0.09811204335886783}, {"donor": 21, "receiver": 20, "amplitude": 0.010575637221336365, "score": 0.07773469150681428}]
- 候选 5: progress=0.9371, opportunity=0.0082, complexity=0.1440, priority=0.8195, support=6, plan=[{"donor": 3, "receiver": 7, "amplitude": 0.010575637221336365, "score": 0.016926266019934362}, {"donor": 18, "receiver": 12, "amplitude": 0.010575637221336365, "score": 0.24018821105135496}, {"donor": 21, "receiver": 15, "amplitude": 0.010575637221336365, "score": 0.19385910677392895}]
- 候选 6: progress=0.9385, opportunity=0.0088, complexity=0.1467, priority=0.8189, support=6, plan=[{"donor": 3, "receiver": 12, "amplitude": 0.010575637221336365, "score": 0.1309794998330114}, {"donor": 7, "receiver": 15, "amplitude": 0.010575637221336365, "score": 0.06550675138380231}, {"donor": 21, "receiver": 20, "amplitude": 0.010575637221336365, "score": 0.07773469150681428}]
- 候选 7: progress=0.9355, opportunity=0.0076, complexity=0.1443, priority=0.8179, support=6, plan=[{"donor": 3, "receiver": 9, "amplitude": 0.010575637221336365, "score": 0.05668832089421669}, {"donor": 6, "receiver": 12, "amplitude": 0.010575637221336365, "score": 0.06525381750616965}, {"donor": 21, "receiver": 15, "amplitude": 0.010575637221336365, "score": 0.19385910677392895}]
- 候选 8: progress=0.9353, opportunity=0.0076, complexity=0.1447, priority=0.8175, support=6, plan=[{"donor": 3, "receiver": 12, "amplitude": 0.010575637221336365, "score": 0.1309794998330114}, {"donor": 19, "receiver": 14, "amplitude": 0.010575637221336365, "score": -0.1736049261882811}, {"donor": 21, "receiver": 15, "amplitude": 0.010575637221336365, "score": 0.19385910677392895}]
- 候选 9: progress=0.9358, opportunity=0.0078, complexity=0.1454, priority=0.8174, support=6, plan=[{"donor": 3, "receiver": 12, "amplitude": 0.010575637221336365, "score": 0.1309794998330114}, {"donor": 19, "receiver": 14, "amplitude": 0.010954543948173523, "score": -0.1736049261882811}, {"donor": 21, "receiver": 15, "amplitude": 0.010575637221336365, "score": 0.19385910677392895}]
- 候选 10: progress=0.9363, opportunity=0.0079, complexity=0.1461, priority=0.8174, support=6, plan=[{"donor": 3, "receiver": 12, "amplitude": 0.010575637221336365, "score": 0.1309794998330114}, {"donor": 19, "receiver": 14, "amplitude": 0.011333446949720383, "score": -0.1736049261882811}, {"donor": 21, "receiver": 15, "amplitude": 0.010575637221336365, "score": 0.19385910677392895}]
- 候选 11: progress=0.9087, opportunity=0.0362, complexity=0.1149, priority=0.8168, support=4, plan=[{"donor": 21, "receiver": 12, "amplitude": 0.010575637221336365, "score": 0.09958871037403483}, {"donor": 23, "receiver": 15, "amplitude": 0.010575637221336365, "score": -0.10051829621418559}]
- 候选 12: progress=0.9366, opportunity=0.0080, complexity=0.1475, priority=0.8166, support=6, plan=[{"donor": 3, "receiver": 12, "amplitude": 0.010575637221336365, "score": 0.1309794998330114}, {"donor": 19, "receiver": 14, "amplitude": 0.012091252952814102, "score": -0.1736049261882811}, {"donor": 21, "receiver": 15, "amplitude": 0.010575637221336365, "score": 0.19385910677392895}]

## 初步结论
- prior graph admissible edges 数量: `437`
- beam completed candidates 数量: `29`
- 本轮只审查算法是否沿显式 target 方向产生合理边和合理搜索展开，不评估真实训练性能。
- 完整 prior graph JSON: `/home/yuhe/clip_dinoiser/artifacts/manual_target_review_k24_seed0/prior_graph_target_v1.json`
- 完整 beam trace JSON: `/home/yuhe/clip_dinoiser/artifacts/manual_target_review_k24_seed0/beam_target_v1_trace.json`
