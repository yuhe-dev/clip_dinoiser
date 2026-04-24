# Debate Design Card

- `experiment_id`: EXP-P1-003
- `phase`: Phase 1
- `loop_kind`: learner_sensitivity_ladder
- `owner`: Auto Proposer
- `budget_tier`: Tier B
- `hypothesis`: If fixed-subset multi-seed noise remains comparable to the global floor, the next priority is to audit whether the learner itself is insufficiently sensitive to composition changes.
- `input_path`: N/A
- `output_dir`: artifacts/research_harness/EXP-P1-003_learner_sensitivity_ladder
- `judge_policy_path`: MISSING

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
