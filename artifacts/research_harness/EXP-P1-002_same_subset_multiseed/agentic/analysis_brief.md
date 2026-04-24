# Analysis Brief

- `experiment_id`: EXP-P1-002
- `phase`: Phase 1
- `loop_kind`: same_subset_multi_seed
- `judge_decision`: promote
- `evidence_level`: E3

## Key Findings

- Multi-seed training noise stdev 0.0089 is meaningfully below the global floor reference 0.0260.
- Fixed-subset training noise is now available as a reusable baseline for later Phase 1 comparisons.

## Verdict

Result is strong enough to advance the current branch.

## Next Hypotheses

- Audit learner sensitivity under broader training regimes because composition effects may still be recoverable.
- Design feature intervention experiments now that training noise is bounded below the global floor.
