# Judge Report

## Basic Info

- Experiment ID: EXP-P1-001
- Decision: promote
- Evidence Level: E3

## Result Summary

- budget_counts: {'1000': 192}
- count: 192
- max: 24.36
- mean: 24.293854166666666
- median: 24.29
- metric_name: mIoU
- min: 24.22
- p10: 24.26
- p90: 24.33
- range: 0.14000000000000057
- sample_experiment_ids: ['rand_subset_s0000_t00', 'rand_subset_s0001_t00', 'rand_subset_s0002_t00', 'rand_subset_s0003_t00', 'rand_subset_s0004_t00']
- source_counts: {'random_subset': 192}
- stdev: 0.02600321332002687
- subset_seed_count: 192
- training_seed_count: 1

## Reasons

- Observed std 0.0260 and range 0.1400 are both below narrow-floor thresholds.
- The current metric spread is tight enough to justify moving to same-subset multi-seed and learner-sensitivity audits.

## Recommended Actions

- Promote EXP-P1-001 as completed and archive this summary as the current global-mIoU floor.
- Run same-subset multi-training-seed experiments to separate subset effect from training noise.
- Run learner sensitivity ladder before expanding downstream search or surrogate work.

## Flags

- protocol_contamination: False
- requires_literature_radar: False
