# Design Pack

- `experiment_id`: EXP-LIT-EXP-P1-004
- `phase`: Phase 1
- `loop_kind`: literature_radar
- `design_class`: executable_candidate
- `objective`: Expand the method space when the current branch is bottlenecked.

## generated_by

agentic_planner_v1

## generated_at_utc

2026-04-24T14:00:24.875466+00:00

## mutation_scope

- `changed`: ['query plan', 'venue priority', 'method ranking']
- `frozen`: ['current phase', 'current bottleneck definition']

## execution_recipe

- build a query plan from stage, failure mode, and modality
- search external literature sources
- rank candidate methods
- emit method cards and reproduction recommendations

## confound_guardrails

- separate infrastructure failures from scientific bottlenecks
- isolate reproduction into design-only lane until admitted

## expected_signal

- `primary`: Produce evidence strong enough to update keep/park/kill for the current branch.
- `secondary`: Produce structured artifacts that can be judged and debated.

## runtime_requirements

- `input_path`: 
- `output_dir`: artifacts/research_harness/EXP-LIT-EXP-P1-004_literature_radar
- `judge_policy_path`: 

## source_paths

- `hypothesis_brief`: 
- `context_packet`: /home/yuhe/slicetune/clip_dinoiser/artifacts/research_harness/EXP-LIT-EXP-P1-004_literature_radar/agentic/context_snapshot.json
