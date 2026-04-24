# Debate Design Card

- `experiment_id`: EXP-P1-004
- `phase`: Phase 1
- `loop_kind`: feature_intervention_matrix
- `owner`: Auto Proposer
- `budget_tier`: Tier B
- `hypothesis`: If the current head-only CLIP-DINOiser learner is too rigid, then at least one more adaptable learner variant will show stronger and more consistent response to matched real probe-axis interventions than to shuffled or matched-random controls.
- `input_path`: artifacts/surrogate_random_v1/manifests/rand_subset_s0145_t00.json
- `output_dir`: artifacts/research_harness/EXP-P1-004_feature_intervention_matrix
- `judge_policy_path`: .slicetune/judge_policies/feature_intervention_matrix_v1.json

## Debate Focus

- 该任务是否符合当前 phase。
- 该任务是否有独立 judge、稳定 artifact、可审计输入。
- 该任务是可执行任务还是 design_only 任务。

## Design Pack Snapshot

- `design_class`: executable_candidate
- `objective`: Design a new research branch under the current phase contract.
- recipe: refine hypothesis
- recipe: define mutation scope
- recipe: freeze evaluation rubric
- recipe: decide whether the branch is design-only or executable

## Rubric Snapshot

- success: Design clarifies the changed variable, frozen comparison, and expected signal.
