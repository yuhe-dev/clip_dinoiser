# Coverage-Aware Materialization Design

## Goal

Improve `slice_remix` materialization so that the realized `1000`-image subset still tracks the searched `target_mixture`, while also reducing systematic loss of key long-tail classes. The beam search objective remains feature-only. The new behavior is added entirely at the subset materialization stage.

## Current Problem

The current system has already been split into two parts:

1. `beam search` finds a target slice mixture using feature-gap reduction.
2. `materialization` turns that target mixture into an actual subset of images for training.

After the recent `quota + mixture repair` fix, realized feature progress now tracks predicted progress much more closely. However, per-class analysis still shows that many classes with large `full50k - baseline` gaps remain missing or near-zero in the realized `1000`-image subsets. This means:

- the search target is no longer the main source of distortion,
- but the final chosen images still under-cover important categories,
- especially classes that random baseline subsets frequently miss.

The design response is to preserve the current feature-based search while making materialization explicitly aware of a small set of important missing classes.

## Design Position

This is **not** a change to:

- prior-graph construction,
- beam-search expansion,
- target-gap computation,
- candidate ranking objective.

It **is** a change to:

- how `sample_budgeted_subset(...)` chooses the final `1000` image ids once a `target_mixture` has already been chosen.

The design therefore keeps the interpretation of `progress` unchanged:

- `progress = (baseline_gap - current_gap) / baseline_gap`

and treats class coverage as a downstream execution constraint rather than a search objective.

## New Materialization Pipeline

The new materializer has three phases.

### Phase 1: Importance weighting

Keep the existing importance-weight computation:

- compute pool-average slice mixture,
- compute per-sample reweighting toward `target_mixture`,
- normalize into selection probabilities.

This phase defines the preferred direction of selection but does not yet guarantee either slice fidelity or class coverage.

### Phase 2: Mixture-preserving subset construction

Keep the existing improved policy:

- per-slice quota allocation from `target_mixture * budget`,
- dominant-slice sampling to satisfy quotas,
- local mixture repair by swap when it lowers `L1(realized_mixture, target_mixture)`.

This phase remains the primary mechanism that makes realized subsets track the searched node.

### Phase 3: Focus-class coverage repair

Add a new post-hoc repair phase that modifies the selected subset only when it improves key class coverage without causing excessive mixture damage.

The phase operates by repeated local swaps:

- swap in one unselected image,
- swap out one selected image,
- accept only if the combined objective improves.

The combined repair score is:

\[
\text{repair\_score} = \text{coverage\_gain} - \alpha \cdot \text{mixture\_damage}
\]

where:

- `coverage_gain` rewards better coverage of important missing classes,
- `mixture_damage` measures how much the swap worsens distance to `target_mixture`,
- `alpha` controls how conservative the repair is.

This creates a clear priority order:

1. stay close to the searched slice mixture,
2. among equally acceptable subsets, prefer ones that cover the important missing classes.

## Focus Classes

The new repair does **not** operate on all `171` COCO-Stuff classes.

Instead it operates on a compact set of `focus classes`, chosen from the classes where:

- `baseline` is weak,
- `full50k` is much stronger.

The default first-version rule is:

1. compute `gap_c = full50k_IoU[c] - baseline_IoU[c]`,
2. keep classes with `gap_c >= tau`,
3. sort by descending `gap_c`,
4. keep the top `K`.

Recommended defaults:

- `tau = 10.0`
- `K = 25`

This ensures the repair budget is spent only on the classes that currently define the main difference between a `1000`-image subset and the `full50k` reference.

## Class Presence Representation

The first version uses **image-level class presence**, not pixel-level class frequency.

For each image:

- read its segmentation mask,
- mark every semantic class that appears at least once,
- store a binary presence vector of shape `[num_classes]`.

This representation is intentionally simple because the goal is not to reproduce full-data class proportions exactly. The goal is to avoid subsets that systematically fail to include key long-tail classes at all.

Image-level presence is sufficient for the first repair stage because the main failure mode currently visible in the experiments is complete or near-complete absence of important classes.

## Coverage Target

The repair phase uses a **soft target**, not a hard quota.

For each focus class, define a small desired minimum count in the realized subset. This count is derived from the full-data reference but clipped into a practical low-budget range.

The first version should avoid trying to match `full50k` exactly. Instead it should only push the selected subset away from pathological under-coverage.

This keeps the repair feasible for `budget = 1000` and avoids destroying mixture fidelity in pursuit of impossible class-frequency replication.

## Swap Logic

Each repair iteration uses:

- current selected indices,
- current coverage counts on focus classes,
- current realized mixture sum.

Candidate swap-in images are preferred if they:

- contain one or more under-covered focus classes,
- improve the current coverage deficit the most,
- do so with minimal mixture distortion.

Candidate swap-out images are preferred if they:

- contribute little to currently under-covered focus classes,
- and cause minimal loss on already-satisfied focus classes,
- while minimizing mixture distortion.

The algorithm should stop when:

- no positive-score swap exists,
- or a fixed repair budget is exhausted.

## Diagnostics

The materializer should emit diagnostics so experiments can distinguish:

- feature-target fidelity,
- class-coverage improvement,
- and the tradeoff between them.

The first version should record:

- mixture distance before and after coverage repair,
- focus-class coverage counts before and after repair,
- number of accepted swaps,
- focus class list and thresholds used.

These diagnostics should be stored with the generated manifest metadata so downstream analysis can relate them to final training performance.

## Interface Changes

The public entrypoint should remain stable for existing callers.

`sample_budgeted_subset(...)` should accept new optional inputs:

- `class_presence`
- `focus_class_indices`
- optional repair configuration parameters

If these are omitted, behavior should fall back to the current mixture-only materializer. This keeps existing experiments and tests valid while allowing recommendation/export codepaths to opt into coverage-aware behavior.

## Validation Plan

The design is successful if the following hold:

1. realized feature progress remains close to predicted progress,
2. focus-class coverage improves relative to the current policy,
3. key `full50k - baseline` gap classes become less likely to remain at zero,
4. the strongest current candidates such as `top05/top06` do not regress in total mIoU and ideally improve further.

The first evaluation should compare:

- old policy,
- mixture-only repaired policy,
- mixture + coverage-repair policy,

while holding fixed:

- the same search trace,
- the same top candidates,
- the same training seed and validation protocol.

## Non-Goals

The first version does not:

- modify beam search scoring,
- modify feature-gap definitions,
- require a new slice-discovery artifact format,
- enforce exact per-class quotas,
- optimize over all classes,
- or attempt a globally optimal combinatorial subset selection.

It is a pragmatic post-search repair layer intended to preserve the existing system structure while directly addressing the empirically observed coverage failure mode.
