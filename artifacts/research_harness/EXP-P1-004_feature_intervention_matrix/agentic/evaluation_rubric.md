# Evaluation Rubric

- `experiment_id`: EXP-P1-004
- `phase`: Phase 1
- `loop_kind`: feature_intervention_matrix
- `primary_metric`: mIoU

## generated_by

agentic_planner_v1

## generated_at_utc

2026-04-24T14:00:24.890761+00:00

## comparison_contract

- `phase`: Phase 1
- `depends_on`: ['EXP-P1-003']
- `design_class`: executable_candidate

## judge_contract

- `contract_type`: learner_adaptability_audit
- `thresholds`: {'minimum_real_axes_with_signal': 1, 'tier_a.minimum_response_to_noise_ratio': 1.0, 'tier_b.minimum_response_to_noise_ratio': 2.0, 'tier_b.minimum_directional_consistency': 0.67, 'tier_b.require_real_beats_shuffled': True, 'tier_b.require_real_beats_random': True, 'maximum_mean_off_target_drift_ratio': 1.0}

## judge_policy_snapshot

- `design_mode`: minimal_learner_adaptability_audit
- `metric_definitions`: {'composition_response_amplitude': 'abs(mIoU(high) - mIoU(low))', 'response_to_noise_ratio': 'composition_response_amplitude / learner_specific_noise_std', 'directional_consistency': 'fraction of seeds whose signed_response matches the majority sign', 'feature_validity_advantage': 'real_feature_guided.response_to_noise_ratio - max(shuffled_feature_guided.response_to_noise_ratio, matched_random_control.response_to_noise_ratio)'}
- `minimum_completed_learner_variants`: 3
- `minimum_probe_axes`: 2
- `minimum_seed_count_per_cell`: 3
- `minimum_control_families`: 3
- `tier_a_requirements`: {'minimum_response_to_noise_ratio': 1.0, 'require_realized_target_delta_logged': True, 'maximum_mean_off_target_drift_ratio': 1.0}
- `screen_to_confirm_rule`: {'minimum_response_to_noise_ratio': 1.0}
- `promote_requirements`: {'minimum_real_axes_with_signal': 1, 'minimum_response_to_noise_ratio': 2.0, 'minimum_directional_consistency': 0.67, 'require_real_beats_shuffled': True, 'require_real_beats_random': True, 'require_full_validation': True, 'require_teacher_frozen': True}
- `caution_requirements`: {'maximum_mean_off_target_drift_ratio': 1.0}

## success_criteria

- At least one learner-axis cell passes Tier B with response_to_noise_ratio >= 2.0, directional_consistency >= 0.67, and real_feature_guided beating both shuffled and matched-random controls.
- Teacher policy remains frozen and validation mode remains full.

## caution_criteria

- Tier A is screening evidence only and cannot by itself justify a strong research claim.
- If off_target_drift_ratio > 1.0, treat the result as weak evidence only.
- If learner response increases together with learner noise, prefer a more controllable learner over a simply stronger one.

## failure_criteria

- All learner-axis cells remain below their own noise floors.
- Real features fail to beat shuffled and matched-random controls.
- Realized intervention fidelity remains too poor to support causal interpretation.

## promote_rule

Promote the current learner-feature coupling branch only when at least one learner-axis cell satisfies the Tier B success criteria under a valid protocol.

## park_rule

Park when there is weak signal but only Tier A evidence or insufficient fidelity/seed depth to justify a strong claim.

## kill_rule

Kill the current probe axis or learner branch when repeated valid execution still leaves real features below both controls and the learner-specific noise floor.

## source_paths

- `context_packet`: /home/yuhe/slicetune/clip_dinoiser/artifacts/research_harness/EXP-P1-004_feature_intervention_matrix/agentic/context_snapshot.json
- `judge_policy_path`: .slicetune/judge_policies/feature_intervention_matrix_v1.json
- `design_spec_path`: .slicetune/experiments/EXP-P1-004_design_spec.md

## metric_definitions

- `signed_response`: mIoU(high) - mIoU(low)
- `composition_response_amplitude`: abs(signed_response)
- `response_to_noise_ratio`: composition_response_amplitude / learner_specific_noise_std
- `directional_consistency`: fraction of seeds whose signed_response matches the majority sign
- `feature_validity_advantage`: real_feature_guided.response_to_noise_ratio - max(shuffled_feature_guided.response_to_noise_ratio, matched_random_control.response_to_noise_ratio)
