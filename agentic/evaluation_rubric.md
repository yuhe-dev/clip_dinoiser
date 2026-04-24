# Evaluation Rubric

- `experiment_id`: EXP-P1-003
- `phase`: Phase 1
- `loop_kind`: learner_sensitivity_ladder
- `primary_metric`: mIoU

## generated_by

agentic_planner_v1

## generated_at_utc

2026-04-13T18:00:58.910737+00:00

## comparison_contract

- `phase`: Phase 1
- `depends_on`: ['EXP-P1-002']
- `design_class`: executable_candidate

## success_criteria

- Design clarifies the changed variable, frozen comparison, and expected signal.

## caution_criteria

- Do not promote without a task-specific evidence floor.

## failure_criteria

- Design remains ambiguous or unexecutable.

## promote_rule

Promote only when success criteria are met and no major protocol contamination is detected.

## park_rule

Park when interesting signal exists but the current phase or evidence depth is insufficient.

## kill_rule

Kill when repeated execution still fails to produce interpretable signal under a valid protocol.

## source_paths

- `context_packet`: /home/yuhe/clip_dinoiser/agentic/context_snapshot.json
- `judge_policy_path`:
