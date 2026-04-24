# Evaluation Rubric

- `experiment_id`: EXP-P1-002
- `phase`: Phase 1
- `loop_kind`: same_subset_multi_seed
- `primary_metric`: mIoU

## generated_by

agentic_planner_v1

## generated_at_utc

2026-04-24T14:00:24.882726+00:00

## comparison_contract

- `phase`: Phase 1
- `depends_on`: []
- `design_class`: executable_candidate

## judge_contract

- `contract_type`: same_subset_multi_seed
- `thresholds`: {'minimum_completed_runs': 3, 'global_floor_stdev_reference': 0.0260032133, 'comparable_noise_ratio': 1.0}

## judge_policy_snapshot

- `minimum_completed_runs`: 3
- `global_floor_stdev_reference`: 0.0260032133
- `comparable_noise_ratio`: 1.0

## success_criteria

- Completed multi-seed count meets the minimum requirement.
- Training noise is meaningfully below the global floor reference.

## caution_criteria

- If noise is comparable to the floor, downstream claims need stronger controls.

## failure_criteria

- Seed runs are incomplete or corrupted.

## promote_rule

Promote only when success criteria are met and no major protocol contamination is detected.

## park_rule

Park when interesting signal exists but the current phase or evidence depth is insufficient.

## kill_rule

Kill when repeated execution still fails to produce interpretable signal under a valid protocol.

## source_paths

- `context_packet`: /home/yuhe/slicetune/clip_dinoiser/artifacts/research_harness/EXP-P1-002_same_subset_multiseed/agentic/context_snapshot.json
- `judge_policy_path`: .slicetune/judge_policies/same_subset_multiseed_v1.json
