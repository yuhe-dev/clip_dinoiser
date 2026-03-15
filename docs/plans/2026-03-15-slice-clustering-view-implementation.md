# Slice Clustering View Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a UMAP-based `Slice Clustering View` to the existing slice report workflow by exporting 2D embedding artifacts from `clip_dinoiser` and rendering them in `client` `View 3`.

**Architecture:** Extend the report export pipeline to generate stable 2D UMAP artifacts from the projected clustering space, then load those artifacts in the `client` slice report store and render them in a new D3-based clustering panel. Reorganize `View 3` so the clustering view becomes the central global overview of the slice workspace.

**Tech Stack:** Python, NumPy, report exporter, Vue 3, Pinia, D3, Vite, TypeScript

---

### Task 1: Add backend report schema for 2D embedding artifacts

**Files:**
- Modify: `clip_dinoiser/slice_discovery/report_exporter.py`
- Modify: `clip_dinoiser/run_slice_report_export.py`
- Test: `clip_dinoiser/tests/test_slice_report_export.py`

**Step 1: Write the failing test**

Add a test asserting that report export produces:

- `embedding_2d.json`
- `slice_centers_2d.json`
- `run_summary.json` with an `embedding` section

**Step 2: Run test to verify it fails**

Run: `python -m unittest clip_dinoiser.tests.test_slice_report_export -v`
Expected: FAIL because the new files and metadata are missing.

**Step 3: Write minimal implementation**

Update the report exporter to reserve output paths and metadata schema for:

- per-sample 2D embedding entries
- per-slice 2D center entries
- run-level embedding metadata

**Step 4: Run test to verify it passes**

Run: `python -m unittest clip_dinoiser.tests.test_slice_report_export -v`
Expected: PASS

**Step 5: Commit**

```bash
git add clip_dinoiser/slice_discovery/report_exporter.py clip_dinoiser/run_slice_report_export.py clip_dinoiser/tests/test_slice_report_export.py
git commit -m "feat: add report schema for slice clustering embedding"
```

### Task 2: Compute and export UMAP 2D sample embedding

**Files:**
- Modify: `clip_dinoiser/slice_discovery/report_exporter.py`
- Test: `clip_dinoiser/tests/test_slice_report_export.py`

**Step 1: Write the failing test**

Add a test asserting that:

- `embedding_2d.json` contains one entry per sample
- each entry includes `sample_id`, `x`, `y`, `hard_assignment`, `max_membership`
- all coordinates are finite numbers

**Step 2: Run test to verify it fails**

Run: `python -m unittest clip_dinoiser.tests.test_slice_report_export -v`
Expected: FAIL because no embedding is computed yet.

**Step 3: Write minimal implementation**

Implement UMAP 2D projection over the projected clustering matrix with fixed parameters and export the per-sample embedding JSON.

**Step 4: Run test to verify it passes**

Run: `python -m unittest clip_dinoiser.tests.test_slice_report_export -v`
Expected: PASS

**Step 5: Commit**

```bash
git add clip_dinoiser/slice_discovery/report_exporter.py clip_dinoiser/tests/test_slice_report_export.py
git commit -m "feat: export sample-level umap embedding for slice reports"
```

### Task 3: Export 2D slice centers and embedding metadata

**Files:**
- Modify: `clip_dinoiser/slice_discovery/report_exporter.py`
- Test: `clip_dinoiser/tests/test_slice_report_export.py`

**Step 1: Write the failing test**

Add assertions that:

- `slice_centers_2d.json` contains one center per slice
- `run_summary.json` includes `embedding.method`, `embedding.random_state`, `embedding.n_neighbors`, and `embedding.min_dist`

**Step 2: Run test to verify it fails**

Run: `python -m unittest clip_dinoiser.tests.test_slice_report_export -v`
Expected: FAIL because center metadata is missing.

**Step 3: Write minimal implementation**

Project slice centers into the 2D UMAP space and write embedding metadata into the run summary.

**Step 4: Run test to verify it passes**

Run: `python -m unittest clip_dinoiser.tests.test_slice_report_export -v`
Expected: PASS

**Step 5: Commit**

```bash
git add clip_dinoiser/slice_discovery/report_exporter.py clip_dinoiser/tests/test_slice_report_export.py
git commit -m "feat: export slice centers and embedding metadata"
```

### Task 4: Extend the frontend slice report store

**Files:**
- Modify: `client/src/stores/sliceReport.ts`
- Test: `client` build validation

**Step 1: Write the failing type updates**

Add TypeScript interfaces for:

- sample embedding records
- slice center embedding records

Then update the store state to include:

- `embedding2d`
- `sliceCenters2d`

**Step 2: Run build to verify it fails**

Run: `cd '/Users/yuhe/research/UIST 2026/client' && npx vite build`
Expected: FAIL until the new fields are loaded and typed consistently.

**Step 3: Write minimal implementation**

Update `loadRun()` so it fetches:

- `/reports/<run_id>/embedding_2d.json`
- `/reports/<run_id>/slice_centers_2d.json`

and stores them in Pinia state.

**Step 4: Run build to verify it passes**

Run: `cd '/Users/yuhe/research/UIST 2026/client' && npx vite build`
Expected: PASS

**Step 5: Commit**

```bash
git add client/src/stores/sliceReport.ts
git commit -m "feat: load slice clustering embedding artifacts in store"
```

### Task 5: Add SliceClusteringView component

**Files:**
- Create: `client/src/views/view3/components/SliceClusteringView.vue`
- Modify: `client/src/stores/sliceReport.ts`
- Test: `client` build validation

**Step 1: Write the component skeleton**

Create a component that accepts store-driven data and renders:

- empty state when no embedding is loaded
- loading-compatible shell
- panel title and plotting region

**Step 2: Run build to verify it fails**

Run: `cd '/Users/yuhe/research/UIST 2026/client' && npx vite build`
Expected: FAIL until imports and props/state usage are completed.

**Step 3: Write minimal implementation**

Use D3 to render:

- sample points at `(x, y)`
- point color by `hard_assignment`
- point opacity by `max_membership`
- slice centers as larger markers
- click on center to call `selectSlice(sliceId)`

Add tooltip support for sample hover.

**Step 4: Run build to verify it passes**

Run: `cd '/Users/yuhe/research/UIST 2026/client' && npx vite build`
Expected: PASS

**Step 5: Commit**

```bash
git add client/src/views/view3/components/SliceClusteringView.vue client/src/stores/sliceReport.ts
git commit -m "feat: add slice clustering scatter view"
```

### Task 6: Reorganize View 3 layout

**Files:**
- Modify: `client/src/views/view3/View3.vue`
- Test: `client` build validation

**Step 1: Update the layout plan**

Rework `View3.vue` so the main layout becomes:

- left: `SliceListView`
- center top: `SliceClusteringView`
- center bottom: `SliceComparisonView`
- right top: `SliceFeatureView`
- right middle: representative samples
- right bottom: boundary samples

**Step 2: Run build to verify it fails if imports/layout are incomplete**

Run: `cd '/Users/yuhe/research/UIST 2026/client' && npx vite build`
Expected: FAIL until layout wiring is correct.

**Step 3: Write minimal implementation**

Import `SliceClusteringView.vue`, insert it into the center column, and rebalance row heights so the clustering panel becomes the main overview surface.

**Step 4: Run build to verify it passes**

Run: `cd '/Users/yuhe/research/UIST 2026/client' && npx vite build`
Expected: PASS

**Step 5: Commit**

```bash
git add client/src/views/view3/View3.vue
git commit -m "feat: promote slice clustering view in workspace layout"
```

### Task 7: Validate report export and frontend loading end-to-end

**Files:**
- Modify: `clip_dinoiser/tests/test_slice_report_export.py`
- Validate: exported report bundle and frontend loading path

**Step 1: Export a local test report**

Run the slice report export command against a small local bundle and confirm the new files are present.

**Step 2: Copy the report into client public reports**

Place the exported test bundle under:

- `client/public/reports/<run_id>/`

**Step 3: Run frontend build and manual smoke check**

Run:

- `cd '/Users/yuhe/research/UIST 2026/client' && npx vite build`

Then manually inspect that the page can load the run id and render the clustering panel.

**Step 4: Record verification**

Document:

- sample count matches between `samples.json` and `embedding_2d.json`
- slice count matches between `num_slices` and `slice_centers_2d.json`
- no empty or invalid coordinate fields

**Step 5: Commit**

```bash
git add clip_dinoiser/tests/test_slice_report_export.py
git commit -m "test: verify end-to-end slice clustering report artifacts"
```

