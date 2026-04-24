# Experiment Progress

- `experiment_id`: EXP-P1-003
- `loop_kind`: learner_sensitivity_ladder
- `current_step`: judge_completed
- `research_state`: judgment
- `next_state`: 
- `updated_at_utc`: 2026-04-13T18:44:34.687598+00:00
- `next_action`: continue_regime_execution
- `acceptance_status`: not_required

## Tasks

- [x] `regime_fast_cached_1ep` [execution] Run learner regime fast_cached_1ep
  - config=feature_experiment_fast_cached_slide mIoU=24.2900
- [x] `regime_fast_1ep` [execution] Run learner regime fast_1ep
  - config=feature_experiment_fast mIoU=20.3900
- [x] `regime_standard_3ep` [execution] Run learner regime standard_3ep
  - config=feature_experiment mIoU=20.7500
- [x] `summarize_results` [verification] Summarize learner-sensitivity ladder
  - completed=3 failed=0
- [x] `judge_results` [judgment] Judge learner-sensitivity signal
  - decision=promote
- [x] `human_acceptance` [acceptance] Human review stop
  - not required

## Recent Facts

- completed_regime_count=3
- failure_count=0
