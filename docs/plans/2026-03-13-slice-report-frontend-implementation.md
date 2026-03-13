# Slice Report Frontend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Export a static slice report bundle from `clip_dinoiser` and render it inside a new `View 3` dashboard in the existing Vue client.

**Architecture:** Keep report semantics in `clip_dinoiser` by exporting a stable JSON + thumbnails bundle, then let `client` load the bundle from `public/reports/<run_id>/`. Build the frontend as a new dashboard workspace with one store and a small set of focused panels rather than a separate app.

**Tech Stack:** Python, NumPy, unittest, Vue 3, Pinia, Ant Design Vue, D3, Vite

---

### Task 1: Define the report schema and exporter entrypoint

**Files:**
- Create: `slice_discovery/report_exporter.py`
- Create: `run_slice_report_export.py`
- Modify: `slice_discovery/__init__.py`
- Test: `tests/test_slice_report_export.py`

**Step 1: Write the failing test**

Add a test that:

- creates a tiny assembled/projected/clustering fixture
- runs the report exporter entrypoint
- expects:
  - `run_summary.json`
  - `slices.json`
  - `samples.json`
  - `feature_schema.json`

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_slice_report_export -v
```

Expected: FAIL because exporter does not exist.

**Step 3: Write minimal implementation**

Implement:

- a report exporter class
- a direct-run script
- stable JSON schema generation

Do not add thumbnail generation yet.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_slice_report_export -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add slice_discovery/report_exporter.py run_slice_report_export.py slice_discovery/__init__.py tests/test_slice_report_export.py
git commit -m "feat: add slice report exporter"
```

### Task 2: Export slice interpretability summaries

**Files:**
- Modify: `slice_discovery/report_exporter.py`
- Test: `tests/test_slice_report_export.py`

**Step 1: Write the failing tests**

Add tests that expect:

- per-slice `top_shifted_features`
- per-slice `block_portrait`
- representative sample ids
- center sample ids
- ambiguous sample ids

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_slice_report_export -v
```

Expected: FAIL for missing interpretability fields.

**Step 3: Write minimal implementation**

Implement:

- global means
- slice means
- shift scoring
- representative / center / ambiguous sample selection

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_slice_report_export -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add slice_discovery/report_exporter.py tests/test_slice_report_export.py
git commit -m "feat: add slice interpretability summaries"
```

### Task 3: Add thumbnail export support

**Files:**
- Modify: `slice_discovery/report_exporter.py`
- Modify: `run_slice_report_export.py`
- Test: `tests/test_slice_report_export.py`

**Step 1: Write the failing test**

Add a test that expects the exporter to create thumbnail asset references for representative samples when given an image root.

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_slice_report_export -v
```

Expected: FAIL because thumbnails are absent.

**Step 3: Write minimal implementation**

Implement:

- thumbnail output directory
- image url fields in `samples.json`
- optional thumbnail generation or passthrough image mapping

Keep the first version simple and deterministic.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_slice_report_export -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add slice_discovery/report_exporter.py run_slice_report_export.py tests/test_slice_report_export.py
git commit -m "feat: export slice thumbnails"
```

### Task 4: Add report store in client

**Files:**
- Create: `../client/src/stores/sliceReport.ts`
- Test: `../client` lightweight manual validation only

**Step 1: Implement the store**

Add a Pinia store that:

- loads `run_summary.json`
- loads `slices.json`
- loads `samples.json`
- tracks selected run and selected slice
- exposes derived getters for the current slice panels

**Step 2: Validate store manually**

Run:

```bash
cd ../client
npm run build
```

Expected: build succeeds

**Step 3: Commit**

```bash
git add ../client/src/stores/sliceReport.ts
git commit -m "feat: add slice report store"
```

### Task 5: Add View 3 dashboard shell

**Files:**
- Modify: `../client/src/App.vue`
- Create: `../client/src/views/view3/View3.vue`
- Create: `../client/src/views/view3/components/SliceListView.vue`
- Create: `../client/src/views/view3/components/SliceFeatureView.vue`
- Create: `../client/src/views/view3/components/SliceComparisonView.vue`
- Create: `../client/src/views/view3/components/SliceSamplesView.vue`
- Create: `../client/src/views/view3/components/SliceBoundaryView.vue`

**Step 1: Implement the view shell**

Add:

- a third top-level view switch button
- a `View3` layout matching the existing dashboard language

**Step 2: Implement placeholder panel wiring**

Each panel should:

- accept the selected slice data
- render basic content cleanly
- handle loading and empty states

**Step 3: Validate manually**

Run:

```bash
cd ../client
npm run build
```

Expected: build succeeds

**Step 4: Commit**

```bash
git add ../client/src/App.vue ../client/src/views/view3
git commit -m "feat: add slice analysis workspace shell"
```

### Task 6: Add feature and image panels

**Files:**
- Modify: `../client/src/views/view3/components/SliceFeatureView.vue`
- Modify: `../client/src/views/view3/components/SliceComparisonView.vue`
- Modify: `../client/src/views/view3/components/SliceSamplesView.vue`
- Modify: `../client/src/views/view3/components/SliceBoundaryView.vue`

**Step 1: Implement feature interpretation panels**

Show:

- top shifted features table
- block portrait bars or heatmap
- global vs slice comparison

**Step 2: Implement image panels**

Show:

- representative image grid
- center image grid
- ambiguous image grid

**Step 3: Validate manually**

Run:

```bash
cd ../client
npm run build
```

Expected: build succeeds

**Step 4: Commit**

```bash
git add ../client/src/views/view3/components
git commit -m "feat: render slice features and sample grids"
```

### Task 7: Add run loading and static bundle integration

**Files:**
- Modify: `../client/src/stores/sliceReport.ts`
- Modify: `../client/src/views/view3/View3.vue`
- Optionally add: `../client/public/reports/<example_run>/...`

**Step 1: Implement static report loading**

Support:

- one configured default run
- loading report files from `/reports/<run_id>/`
- loading state and error state

**Step 2: Validate against a real exported bundle**

Run:

```bash
python run_slice_report_export.py ...
cp -R artifacts/slice_report/<run_id> ../client/public/reports/<run_id>
cd ../client
npm run build
```

Expected: the client can load and render the report

**Step 3: Commit**

```bash
git add ../client/src/stores/sliceReport.ts ../client/src/views/view3/View3.vue
git commit -m "feat: load static slice report bundles"
```

### Task 8: Final verification

**Files:**
- Modify as needed based on verification findings

**Step 1: Run Python verification**

Run:

```bash
python -m unittest tests.test_processed_feature_assembler tests.test_slice_feature_projector tests.test_slice_finder tests.test_slice_baseline_cli tests.test_slice_debug_scripts tests.test_slice_report_export -v
```

Expected: PASS

**Step 2: Run frontend verification**

Run:

```bash
cd ../client
npm run build
```

Expected: PASS

**Step 3: Smoke test with a real run**

Run:

```bash
python run_slice_report_export.py ...
cp -R artifacts/slice_report/<run_id> ../client/public/reports/<run_id>
cd ../client
npm run dev
```

Verify:

- View 3 loads
- slice list renders
- feature panel renders
- image grids render

**Step 4: Commit**

```bash
git add .
git commit -m "feat: integrate slice report frontend workspace"
```
