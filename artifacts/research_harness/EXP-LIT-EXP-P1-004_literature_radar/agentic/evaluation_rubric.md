# Evaluation Rubric

- `experiment_id`: EXP-LIT-EXP-P1-004
- `phase`: Phase 1
- `loop_kind`: literature_radar
- `primary_metric`: mIoU

## generated_by

agentic_planner_v1

## generated_at_utc

2026-04-24T14:00:24.875473+00:00

## comparison_contract

- `phase`: Phase 1
- `depends_on`: ['EXP-P1-004']
- `design_class`: executable_candidate

## judge_contract

- `contract_type`: literature_radar
- `thresholds`: {'minimum_ranked_results': 6, 'minimum_reproduce_candidates': 1, 'maximum_search_error_ratio': 0.5}

## success_criteria

- At least several methods are retrieved, ranked, and scoped to the current bottleneck.
- At least one method card is recommended for reproduction or deeper study.

## caution_criteria

- Avoid over-expanding into unrelated adjacent research areas.

## failure_criteria

- Search only returns low-relevance or duplicate methods.

## promote_rule

Promote only when success criteria are met and no major protocol contamination is detected.

## park_rule

Park when interesting signal exists but the current phase or evidence depth is insufficient.

## kill_rule

Kill when repeated execution still fails to produce interpretable signal under a valid protocol.

## source_paths

- `context_packet`: /home/yuhe/slicetune/clip_dinoiser/artifacts/research_harness/EXP-LIT-EXP-P1-004_literature_radar/agentic/context_snapshot.json
- `judge_policy_path`:
