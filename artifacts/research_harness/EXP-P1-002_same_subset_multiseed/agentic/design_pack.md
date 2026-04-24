# Design Pack

- `experiment_id`: EXP-P1-002
- `phase`: Phase 1
- `loop_kind`: same_subset_multi_seed
- `design_class`: executable_candidate
- `objective`: Estimate training noise while holding subset composition fixed.

## generated_by

agentic_planner_v1

## generated_at_utc

2026-04-24T14:00:24.882637+00:00

## mutation_scope

- `changed`: ['training_seed']
- `frozen`: ['subset manifest', 'config', 'metric', 'budget', 'runtime profile']

## execution_recipe

- materialize per-seed manifests from a fixed subset
- launch one training run per seed
- reuse completed runs when provenance matches
- summarize seed-wise metric spread

## confound_guardrails

- fixed subset manifest across all seeds
- matched runtime profile and config
- completion sentinel must match provenance

## expected_signal

- `primary`: Produce evidence strong enough to update keep/park/kill for the current branch.
- `secondary`: Produce structured artifacts that can be judged and debated.

## runtime_requirements

- `input_path`: artifacts/surrogate_random_v1/manifests/rand_subset_s0145_t00.json
- `output_dir`: artifacts/research_harness/EXP-P1-002_same_subset_multiseed
- `judge_policy_path`: .slicetune/judge_policies/same_subset_multiseed_v1.json

## source_paths

- `hypothesis_brief`: 
- `context_packet`: /home/yuhe/slicetune/clip_dinoiser/artifacts/research_harness/EXP-P1-002_same_subset_multiseed/agentic/context_snapshot.json
