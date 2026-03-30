# Coverage-Aware Materialization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a coverage-aware post-processing phase to `slice_remix` materialization so realized subsets stay close to `target_mixture` while improving coverage of the highest-gap classes from `full50k - baseline`.

**Architecture:** Keep beam search and feature-gap scoring unchanged. Extend `sample_budgeted_subset(...)` with optional class-presence inputs and run a post-hoc swap repair after the existing quota-based mixture repair. Add a small CLI utility layer in recommendation/export paths to construct focus-class metadata and image-level class presence from COCO-Stuff annotation masks.

**Tech Stack:** Python 3, NumPy, standard library JSON/path utilities, existing `slice_remix` package, `unittest`

---

### Task 1: Add coverage-aware policy unit tests

**Files:**
- Modify: `clip_dinoiser/tests/test_slice_remix_policy.py`

**Step 1: Write the failing tests**

Add tests that cover:

- `sample_budgeted_subset(...)` still works in mixture-only mode,
- coverage-aware mode increases focus-class coverage on a toy dataset,
- coverage-aware mode does not explode mixture distance relative to the pre-repair selection.

**Step 2: Run tests to verify failure**

Run:

```bash
python3 -m unittest clip_dinoiser.tests.test_slice_remix_policy -v
```

Expected: FAIL because coverage-aware helpers do not exist yet.

**Step 3: Implement minimal failing fixtures**

Use a small synthetic dataset:

- 6-10 samples,
- 2-3 slices,
- 3-4 semantic classes,
- one or two focus classes absent from the initial mixture-only selection.

**Step 4: Run tests again**

Run:

```bash
python3 -m unittest clip_dinoiser.tests.test_slice_remix_policy -v
```

Expected: targeted coverage-aware tests still fail, existing tests still run.

**Step 5: Commit**

```bash
git add clip_dinoiser/tests/test_slice_remix_policy.py
git commit -m "test: add coverage-aware materialization cases"
```

### Task 2: Extend `policy.py` with focus-class repair helpers

**Files:**
- Modify: `clip_dinoiser/slice_remix/policy.py`
- Test: `clip_dinoiser/tests/test_slice_remix_policy.py`

**Step 1: Add helper functions**

Implement:

- `build_focus_class_targets(...)`
- `_focus_class_coverage_counts(...)`
- `_coverage_gain(...)`
- `_coverage_repair_subset(...)`

Keep them internal except for any function that genuinely needs reuse.

**Step 2: Update `sample_budgeted_subset(...)`**

Extend the signature with optional arguments:

- `class_presence: np.ndarray | None = None`
- `focus_class_indices: list[int] | None = None`
- optional repair config parameters such as `coverage_alpha`, `coverage_repair_budget`

Behavior:

- if coverage inputs are absent, keep current behavior unchanged,
- if present, run the new coverage-repair phase after the current mixture repair.

**Step 3: Preserve backward compatibility**

Make sure:

- existing callers that only pass `weights`, `memberships`, and `target_mixture` still work,
- returned sample id format is unchanged,
- current mixture-repair logic stays the default backbone.

**Step 4: Run policy tests**

Run:

```bash
python3 -m unittest clip_dinoiser.tests.test_slice_remix_policy -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add clip_dinoiser/slice_remix/policy.py clip_dinoiser/tests/test_slice_remix_policy.py
git commit -m "feat: add coverage-aware post-hoc materialization repair"
```

### Task 3: Add mask-derived class-presence utilities

**Files:**
- Create: `clip_dinoiser/slice_remix/class_coverage.py`
- Create: `clip_dinoiser/tests/test_slice_remix_class_coverage.py`

**Step 1: Write the failing tests**

Cover:

- mapping from `images/train2017/xxx.jpg` to `annotations/train2017/xxx_labelTrainIds.png`,
- loading a tiny grayscale label mask and converting it to image-level class presence,
- selecting top focus classes from baseline/full per-class IoU dictionaries.

**Step 2: Run tests to verify failure**

Run:

```bash
python3 -m unittest clip_dinoiser.tests.test_slice_remix_class_coverage -v
```

Expected: FAIL because the module does not exist.

**Step 3: Implement utilities**

Implement:

- `annotation_path_from_sample_id(...)`
- `load_class_presence_matrix(...)`
- `select_focus_class_indices(...)`

Use the existing COCO-Stuff `labelTrainIds` convention and ignore label `255`.

**Step 4: Run tests**

Run:

```bash
python3 -m unittest clip_dinoiser.tests.test_slice_remix_class_coverage -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add clip_dinoiser/slice_remix/class_coverage.py clip_dinoiser/tests/test_slice_remix_class_coverage.py
git commit -m "feat: add class coverage utilities for materialization"
```

### Task 4: Thread coverage-aware inputs into recommendation/export flows

**Files:**
- Modify: `clip_dinoiser/run_remix_response_dataset.py`
- Modify: `clip_dinoiser/run_remix_validate_recommendation.py`
- Test: `clip_dinoiser/tests/test_remix_response_dataset_cli.py`
- Test: `clip_dinoiser/tests/test_remix_validate_recommendation_cli.py`

**Step 1: Add coverage metadata loading**

In the CLI paths that materialize candidate subsets:

- load or derive `class_presence`,
- compute `focus_class_indices` from baseline/full per-class stats when available,
- pass both into `sample_budgeted_subset(...)`.

If the required per-class reference inputs are absent, log and fall back to mixture-only materialization.

**Step 2: Persist diagnostics**

Attach to manifest metadata:

- focus class ids/names,
- accepted swap count,
- mixture distance before/after coverage repair,
- coverage counts before/after repair.

**Step 3: Add or update CLI tests**

Use lightweight stubs/mocks so tests only validate:

- optional coverage path is invoked when inputs exist,
- fallback path still works when inputs do not.

**Step 4: Run targeted CLI tests**

Run:

```bash
python3 -m unittest \
  clip_dinoiser.tests.test_remix_response_dataset_cli \
  clip_dinoiser.tests.test_remix_validate_recommendation_cli -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add \
  clip_dinoiser/run_remix_response_dataset.py \
  clip_dinoiser/run_remix_validate_recommendation.py \
  clip_dinoiser/tests/test_remix_response_dataset_cli.py \
  clip_dinoiser/tests/test_remix_validate_recommendation_cli.py
git commit -m "feat: thread coverage-aware materialization into remix CLIs"
```

### Task 5: Add diagnostics for experiment comparison

**Files:**
- Modify: `clip_dinoiser/run_remix_response_dataset.py`
- Modify: `clip_dinoiser/run_workbench_bundle_export.py` if useful for downstream display
- Test: `clip_dinoiser/tests/test_workbench_bundle_export_cli.py` only if touched

**Step 1: Export manifest-side diagnostics**

Make sure generated candidate manifests include structured metadata for:

- `materialization_policy`,
- `focus_classes`,
- `coverage_repair_budget`,
- `coverage_alpha`,
- `mixture_l1_before_coverage_repair`,
- `mixture_l1_after_coverage_repair`,
- `focus_coverage_before`,
- `focus_coverage_after`.

**Step 2: Keep optional behavior**

Do not require frontend/workbench code to consume these fields immediately. They are for offline diagnosis first.

**Step 3: Run relevant tests**

Run the smallest affected suite, for example:

```bash
python3 -m unittest \
  clip_dinoiser.tests.test_remix_response_dataset_cli \
  clip_dinoiser.tests.test_workbench_bundle_export_cli -v
```

Expected: PASS

**Step 4: Commit**

```bash
git add clip_dinoiser/run_remix_response_dataset.py clip_dinoiser/run_workbench_bundle_export.py
git commit -m "feat: export coverage-aware materialization diagnostics"
```

### Task 6: Run local verification suite

**Files:**
- No code changes expected

**Step 1: Run focused unit tests**

Run:

```bash
python3 -m unittest \
  clip_dinoiser.tests.test_slice_remix_policy \
  clip_dinoiser.tests.test_slice_remix_class_coverage \
  clip_dinoiser.tests.test_remix_response_dataset_cli \
  clip_dinoiser.tests.test_remix_validate_recommendation_cli \
  clip_dinoiser.tests.test_validation_acceleration -v
```

Expected: PASS

**Step 2: Smoke test manifest generation**

Run a small-scale local invocation or fixture-backed CLI test that produces:

- one baseline manifest,
- one candidate manifest,
- coverage-aware diagnostics in the output JSON.

Expected: generated manifest includes both mixture and coverage diagnostics.

**Step 3: Commit**

```bash
git add -A
git commit -m "test: verify coverage-aware materialization flow"
```

### Task 7: Remote experiment protocol

**Files:**
- No code changes required

**Step 1: Regenerate manifests on server**

Regenerate:

- baseline
- top01-top10

using the new coverage-aware materializer.

**Step 2: Re-run highest-value candidates first**

Run:

- `baseline`
- `top05`
- `top06`

before re-running the full top10.

**Step 3: Compare three policy generations**

Build a comparison table for:

- old policy,
- quota+mixture-repair policy,
- quota+mixture-repair+coverage-repair policy.

Track:

- total `mIoU`,
- realized progress,
- focus-class coverage counts,
- per-class IoU on the focus classes.

**Step 4: Decide next branch**

If results improve:

- keep search objective fixed,
- continue tuning coverage-repair hyperparameters.

If results do not improve:

- conclude the bottleneck has moved upstream into the feature objective itself.
