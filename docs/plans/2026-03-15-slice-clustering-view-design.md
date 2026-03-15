# Slice Clustering View Design

## Goal

Add a `Slice Clustering View` to the existing `client` `View 3` workspace so users can inspect the global slice structure in a 2D embedding of the clustering space, understand where slices separate or overlap, and drive the rest of the explanation workflow from that global view.

## Why This View Exists

The current slice report page already shows slice summaries, shifted features, and representative samples, but it lacks a global structural view of how samples are distributed in the clustering space. This makes it hard to answer questions such as:

- whether discovered slices form separated regions or overlap heavily
- which slices occupy dense or sparse regions of the projected clustering space
- where high-confidence versus boundary samples lie
- how representative and ambiguous samples relate to the global slice geometry

The new view should therefore act as the primary navigation surface for the slice workspace, not just as an extra chart.

## Visualization Semantics

The view must be described and implemented as:

`a 2D UMAP projection of the clustering space`

It must **not** be framed as the true decision plane of the clustering algorithm. The underlying clustering still happens in the high-dimensional projected feature space. The 2D plot is a visualization-only projection used to expose global structure.

## Recommended Architecture

The 2D embedding should be computed in `clip_dinoiser` during report export, then written into the static report bundle. The `client` should only load and render the embedding. This keeps the system aligned with the existing architecture:

- `clip_dinoiser` computes report artifacts
- `client` loads static report artifacts and renders them

This design is preferable to computing UMAP in the browser because it is:

- reproducible across runs and sessions
- simpler for the frontend
- more appropriate for paper-quality analysis artifacts
- easier to document and verify

## Report Bundle Additions

The report bundle should add:

- `embedding_2d.json`
- `slice_centers_2d.json`

And `run_summary.json` should add:

- `embedding.method`
- `embedding.random_state`
- `embedding.n_neighbors`
- `embedding.min_dist`

### `embedding_2d.json`

Each sample record should contain:

- `sample_id`
- `x`
- `y`
- `hard_assignment`
- `max_membership`

This is the minimum data needed to render the scatter plot while still allowing the frontend to join the points back to `samples.json`.

### `slice_centers_2d.json`

Each slice record should contain:

- `slice_id`
- `x`
- `y`
- `weight`
- `hard_count`

These centers provide explicit anchors for slice-level selection and interpretation.

## Embedding Method

The first implementation should use:

- `UMAP`
- `n_components = 2`
- a fixed `random_state`

The input to UMAP should be the same matrix used by clustering, i.e. the projected clustering-space matrix.

The embedding should be treated as a report artifact, not an online computation.

## View 3 Layout Update

`View 3` should be reorganized into:

- Left column: `Slice List View`
- Center top: `Slice Clustering View`
- Center bottom: `Slice Comparison View`
- Right top: `Slice Feature View`
- Right middle: `Slice Representative View`
- Right bottom: `Slice Boundary View`

This creates a more coherent analysis flow:

1. inspect global slice structure
2. inspect block-level slice contrast
3. inspect feature-level explanation
4. inspect representative and ambiguous examples

## Interaction Model

The first version should support the following interactions:

### Default Global State

- show all samples
- color points by `hard_assignment`
- modulate point opacity by `max_membership`
- display all slice centers
- highlight the currently selected slice while keeping the rest visible at lower emphasis

### Slice Selection

- clicking a slice center should select that slice
- selecting a slice from the left-hand slice list should highlight its points in the scatter

### Sample Inspection

- hovering a point should show a tooltip with:
  - `sample_id`
  - `hard_assignment`
  - `max_membership`
  - top-ranked slices

The first version does not need lasso selection, density contours, or advanced brushing.

## Visual Encoding

The view should stay consistent with the current analytical dashboard aesthetic:

- clean white background
- explicit panel borders
- restrained palette
- technical rather than decorative styling

Recommended encodings:

- point color: `hard_assignment`
- point opacity: `max_membership`
- center marker: larger outlined marker or cross
- minimal axes or grid
- chart title explicitly indicating that this is a UMAP projection

The view should avoid any hard region tessellation or Voronoi-style partition graphics, because those would visually imply a decision boundary that does not exist in the 2D plane.

## Frontend Integration

The `client` should:

- extend the slice report store to load the new embedding files
- add a new `SliceClusteringView.vue` component
- render the scatter plot with D3
- reuse the existing selected-slice store state for linked highlighting

The component should be resilient to missing embedding files and display a clear empty/error state if the selected report bundle does not contain them.

## Validation Requirements

The backend must validate:

- sample count in `embedding_2d.json` matches `samples.json`
- slice center count matches `num_slices`
- all coordinates are finite

The frontend must validate:

- embedding files can be loaded from `/reports/<run_id>/`
- missing embedding files produce a visible empty state
- selecting a slice updates the scatter highlight state
- the page still builds successfully with Vite

## Non-Goals for the First Version

The first version should not include:

- browser-side UMAP computation
- multiple embedding methods
- lasso or brush selection
- animated transitions across runs
- density contours or soft region fields
- cross-run comparison inside the same clustering panel

These can be added later after the base global-structure view is working and readable.

## Summary

The recommended solution is to make `Slice Clustering View` the global structural entry point of the slice workspace by exporting a stable 2D UMAP embedding from `clip_dinoiser` and rendering it inside `client` `View 3` with linked slice selection and confidence-aware visual encoding.

