# Judge Report

## Basic Info

- Experiment ID: EXP-P1-003
- Decision: promote
- Evidence Level: E2

## Result Summary

- base_candidate_id: rand_subset_s0145_t00
- baseline_regime_id: fast_cached_1ep
- best_config_name: feature_experiment_fast_cached_slide
- best_minus_baseline: 0.0
- best_regime_id: fast_cached_1ep
- completed_regime_count: 3
- count: 3
- failure_count: 0
- max: 24.29
- mean: 21.810000000000002
- meaningful_sensitivity_threshold: 0.01341640785
- median: 20.75
- metric_name: mIoU
- min: 20.39
- per_regime_metrics: {'fast_cached_1ep': 24.29, 'fast_1ep': 20.39, 'standard_3ep': 20.75}
- range: 3.8999999999999986
- regime_ids: ['fast_cached_1ep', 'fast_1ep', 'standard_3ep']
- regime_range: 3.8999999999999986
- requested_regime_count: 3
- stdev: 2.1552726045676907
- worst_regime_id: fast_1ep

## Reasons

- Observed learner-regime range 3.9000 is above the meaningful sensitivity threshold 0.0134.
- Best regime `fast_cached_1ep` outperformed baseline `fast_cached_1ep` by 0.0000.
- Rubric contract `learner_sensitivity_ladder` alignment=True.
- Context research_state=audit.

## Recommended Actions

- Promote the strongest learner regime as the next anchor for feature intervention experiments.
- Use this ladder result to decide whether protocol sensitivity or feature choice is the main bottleneck.
- Carry this result forward as a context-aware promoted artifact for the current branch.

## Flags

- protocol_contamination: False
- requires_literature_radar: False
