# Judge Report

## Basic Info

- Experiment ID: EXP-P1-002
- Decision: promote
- Evidence Level: E3

## Result Summary

- base_candidate_id: rand_subset_s0145_t00
- completed_seed_count: 5
- count: 5
- failure_count: 0
- max: 24.29
- mean: 24.285999999999998
- median: 24.29
- metric_name: mIoU
- min: 24.27
- noise_to_global_floor_ratio: 0.34396794760742005
- per_seed_metrics: {'0': 24.29, '1': 24.29, '2': 24.29, '3': 24.29, '4': 24.27}
- range: 0.019999999999999574
- requested_seed_count: 5
- stdev: 0.008944271909998969
- training_seed_values: [0, 1, 2, 3, 4]

## Reasons

- Multi-seed training noise stdev 0.0089 is meaningfully below the global floor reference 0.0260.

## Recommended Actions

- Proceed to feature intervention and slice-leverage audits because data-composition effects may still be separable from training noise.
- Keep this summary as the training-noise baseline for future comparisons.

## Flags

- protocol_contamination: False
- requires_literature_radar: False
