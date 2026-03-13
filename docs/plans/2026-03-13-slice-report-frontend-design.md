# Slice Report Frontend Integration Design

**Date:** 2026-03-13

**Goal:** Visualize slice discovery results and slice-level interpretability in the existing Vue dashboard so users can judge clustering quality and slice semantics by both statistics and images.

## Context

The current `client` app is a Vue 3 + Pinia + Ant Design Vue + D3 analysis dashboard with a multi-panel workspace layout. It already follows a clear “analysis tool” visual language:

- multiple simultaneous panels
- sparse color usage
- white background, strong borders, compact information density
- store-driven data loading

The current `clip_dinoiser` pipeline can already produce:

- assembled features
- projected features
- slice clustering results
- cluster diagnostics

However, these artifacts are not directly usable for visual judgment. They answer whether clustering ran, not whether slices are meaningful, stable, or visually interpretable.

## Product Direction

The slice report should not be built as a separate story-style landing page. It should be integrated as a new dashboard workspace, consistent with the existing system style.

The new experience is:

`View 3: Slice Analysis Workspace`

This workspace should support two simultaneous tasks:

1. judge slice quality numerically
2. inspect slice semantics visually through representative images

## Chosen Data Flow

### Recommendation

Use a static report bundle exported by `clip_dinoiser`, then let `client` load it from `public/reports/<run_id>/`.

### Why

- simplest integration path
- no new backend service required
- stable for experiment reproduction
- easy to iterate on report schema
- later can be upgraded to an API without changing the frontend schema

### Data Flow

1. `clip_dinoiser` exports a slice report bundle for one run
2. the bundle is copied or synced into `client/public/reports/<run_id>/`
3. the Vue frontend loads report JSON with `fetch`
4. thumbnails are served as static assets from the same report directory

## Report Bundle Schema

Each run should export a directory like:

`artifacts/slice_report/gmm_k8_seed0/`

with at least:

- `run_summary.json`
- `slices.json`
- `samples.json`
- `feature_schema.json`
- `thumbnails/`

### `run_summary.json`

Global run information for top-level UI state.

Suggested fields:

- `run_id`
- `finder`
- `num_slices`
- `sample_count`
- `block_order`
- `block_ranges`
- `slice_weights`
- `hard_assignment_counts`
- `membership_health`
  - row sum min/max
  - average max membership
  - average entropy

### `slices.json`

Per-slice report content.

Each slice should include:

- `slice_id`
- `weight`
- `hard_count`
- `avg_max_membership`
- `avg_entropy`
- `top_shifted_features`
  - `block`
  - `field`
  - `slice_mean`
  - `global_mean`
  - `shift_score`
- `block_portrait`
  - one summary per block
  - `slice_mean`
  - `global_mean`
  - `delta`
- `representative_samples`
- `center_samples`
- `ambiguous_samples`

### `samples.json`

Lightweight sample-level metadata for image grids and detail panels.

Suggested fields:

- `sample_id`
- `image_rel`
- `image_url`
- `annotation_url` optional
- `hard_assignment`
- `max_membership`
- `membership_vector`
- `slice_rankings`

### `feature_schema.json`

Human-readable field metadata for display labels.

Suggested fields:

- block name
- feature name
- field name
- label
- type

## Image Strategy

The frontend should not depend on direct access to the full raw image directory.

### Recommendation

Export thumbnails for the subset of samples used in the report:

- representative samples
- center samples
- ambiguous samples

Store them in:

`public/reports/<run_id>/thumbnails/...`

### Why

- much faster page load
- avoids path and permission coupling to the raw dataset
- enough for interpretability browsing
- easier to package and share with collaborators

## Frontend Architecture

Add a third dashboard view:

- `src/views/view3/View3.vue`

Add a dedicated store:

- `src/stores/sliceReport.ts`

Add focused components:

- `src/views/view3/components/SliceListView.vue`
- `src/views/view3/components/SliceFeatureView.vue`
- `src/views/view3/components/SliceComparisonView.vue`
- `src/views/view3/components/SliceSamplesView.vue`
- `src/views/view3/components/SliceBoundaryView.vue`

The store should:

- load report files for the current run
- track the selected slice
- expose derived data for each panel
- manage loading and error states

## View Layout

The layout should remain consistent with the current dashboard style.

### Left Panel: Slice List View

Purpose:

- browse all slices quickly
- sort and select a slice

Show:

- slice id
- weight
- hard count
- average confidence
- average entropy
- short top-feature tags

### Center Top: Slice Feature View

Purpose:

- explain what makes the selected slice distinct

Show:

- top shifted features table
- per-block delta bars
- summary metrics

### Center Bottom: Slice Comparison View

Purpose:

- compare the selected slice against global data or another slice

Show:

- block portrait heatmap
- delta-to-global chart
- optional pairwise comparison mode

### Right Top: Representative Sample View

Purpose:

- show what the slice visually looks like

Show:

- top membership samples
- near-center samples
- membership badge on each image

### Right Bottom: Boundary Sample View

Purpose:

- inspect whether the slice boundary is clean

Show:

- ambiguous samples
- low-confidence samples
- optional membership distribution preview per sample

## Visual Language

The current app already has a usable visual system. The slice report should preserve it.

### Keep

- multi-panel workspace
- strong section borders
- restrained grayscale base
- Ant Design primitives

### Add Carefully

- consistent per-slice accent colors
- compact image cards
- simple heatmaps and bar charts
- clearer panel headers and run metadata strip

### Avoid

- landing-page styling
- oversized hero sections
- over-saturated palettes
- decorative animations

## Responsibilities Split

### `clip_dinoiser`

Responsible for:

- clustering output
- interpretability feature computation
- representative / center / ambiguous sample selection
- thumbnail export
- final report JSON generation

### `client`

Responsible for:

- loading a chosen report bundle
- interactive exploration
- slice selection and comparison
- image and feature visualization

This keeps all data semantics in the research pipeline and all interaction logic in the frontend.

## Phased Rollout

### Phase 1

- single-run static report
- one report directory loaded at a time
- basic slice list + feature view + image grids

### Phase 2

- run selector
- compare multiple runs
- compare multiple slice configurations

### Phase 3

- optional backend API
- recommendation and remix overlays

## Recommendation

Start with:

1. `clip_dinoiser` slice report exporter
2. static bundle sync into `client/public/reports/<run_id>/`
3. `View 3` dashboard integration

This gives immediate interpretability value without introducing backend complexity too early.
