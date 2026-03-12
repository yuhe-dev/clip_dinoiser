# Slice Debug Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build three direct-run debug scripts that persist intermediate slice discovery artifacts and validation summaries for assembler, projector, and clustering.

**Architecture:** Keep the existing slice discovery package as the source of truth, and add thin debug entrypoints that run one layer each and save `npz + json` artifacts. Add only the minimum package changes needed to expose layer-level metadata and iterative clustering diagnostics without coupling debugging to the normal baseline path.

**Tech Stack:** Python, NumPy, unittest, existing `slice_discovery` package

---

### Task 1: Add reusable debug-summary helpers

**Files:**
- Modify: `slice_discovery/assembler.py`
- Modify: `slice_discovery/projector.py`
- Modify: `slice_discovery/types.py`
- Test: `tests/test_processed_feature_assembler.py`
- Test: `tests/test_slice_feature_projector.py`

**Step 1: Write the failing tests**

Add tests that expect:

- assembler metadata/debug summary can report per-block finite/min/max/mean
- projector output can expose per-block post-projection stats

**Step 2: Run tests to verify they fail**

Run:

```bash
python -m unittest tests.test_processed_feature_assembler tests.test_slice_feature_projector -v
```

Expected: failures for missing debug-summary helpers or metadata fields.

**Step 3: Write minimal implementation**

Add:

- assembler helper to summarize blocks and flat view
- projector helper to summarize projected blocks using `block_ranges`
- any small type additions needed for structured debug data

**Step 4: Run tests to verify they pass**

Run:

```bash
python -m unittest tests.test_processed_feature_assembler tests.test_slice_feature_projector -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add slice_discovery/assembler.py slice_discovery/projector.py slice_discovery/types.py tests/test_processed_feature_assembler.py tests/test_slice_feature_projector.py
git commit -m "feat: add slice debug summaries"
```

### Task 2: Add clustering diagnostics output

**Files:**
- Modify: `slice_discovery/finder.py`
- Modify: `slice_discovery/types.py`
- Test: `tests/test_slice_finder.py`

**Step 1: Write the failing tests**

Add tests that expect:

- `SliceFindingResult` can carry optional diagnostics
- soft k-means and GMM return row-sum/iteration-friendly diagnostics without changing membership behavior

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_slice_finder -v
```

Expected: FAIL because diagnostics fields are absent.

**Step 3: Write minimal implementation**

Add optional diagnostics payloads to clustering results:

- input shape
- finite checks
- row-sum min/max
- optional per-iteration log-likelihood for GMM

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_slice_finder -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add slice_discovery/finder.py slice_discovery/types.py tests/test_slice_finder.py
git commit -m "feat: add slice clustering diagnostics"
```

### Task 3: Add assembler debug script

**Files:**
- Create: `run_slice_assembler_debug.py`
- Test: `tests/test_slice_debug_scripts.py`

**Step 1: Write the failing test**

Add a test that:

- creates a fixture bundle
- runs assembler debug script entrypoint
- expects `assembled_features.npz`, `assembled_features_meta.json`, and `assembler_debug.json`

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_slice_debug_scripts.SliceDebugScriptTests.test_assembler_debug_script_writes_artifacts -v
```

Expected: FAIL because script does not exist or output is missing.

**Step 3: Write minimal implementation**

Implement a direct-run script that:

- loads processed bundles
- optionally limits samples
- saves assembled artifact via assembler
- writes debug summary JSON

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_slice_debug_scripts.SliceDebugScriptTests.test_assembler_debug_script_writes_artifacts -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add run_slice_assembler_debug.py tests/test_slice_debug_scripts.py
git commit -m "feat: add assembler debug script"
```

### Task 4: Add projector debug script

**Files:**
- Create: `run_slice_projector_debug.py`
- Test: `tests/test_slice_debug_scripts.py`

**Step 1: Write the failing test**

Add a test that:

- uses assembler debug artifact as input
- runs projector debug script entrypoint
- expects `projected_features.npz`, `projected_features_meta.json`, and `projector_debug.json`

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_slice_debug_scripts.SliceDebugScriptTests.test_projector_debug_script_writes_artifacts -v
```

Expected: FAIL because script does not exist or artifact is missing.

**Step 3: Write minimal implementation**

Implement a direct-run script that:

- loads assembled artifact
- runs projector
- saves projected matrix + metadata + debug summary

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_slice_debug_scripts.SliceDebugScriptTests.test_projector_debug_script_writes_artifacts -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add run_slice_projector_debug.py tests/test_slice_debug_scripts.py
git commit -m "feat: add projector debug script"
```

### Task 5: Add cluster debug script

**Files:**
- Create: `run_slice_cluster_debug.py`
- Test: `tests/test_slice_debug_scripts.py`

**Step 1: Write the failing test**

Add tests that:

- use projector artifact as input
- run cluster debug script with `soft_kmeans`
- run cluster debug script with `gmm`
- expect `slice_result.npz`, `slice_result_meta.json`, and `cluster_debug.json`

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_slice_debug_scripts.SliceDebugScriptTests.test_cluster_debug_script_writes_artifacts_for_soft_kmeans tests.test_slice_debug_scripts.SliceDebugScriptTests.test_cluster_debug_script_writes_artifacts_for_gmm -v
```

Expected: FAIL because script does not exist or diagnostics are missing.

**Step 3: Write minimal implementation**

Implement a direct-run script that:

- loads projected artifact
- runs finder
- saves result artifact and debug JSON

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_slice_debug_scripts.SliceDebugScriptTests.test_cluster_debug_script_writes_artifacts_for_soft_kmeans tests.test_slice_debug_scripts.SliceDebugScriptTests.test_cluster_debug_script_writes_artifacts_for_gmm -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add run_slice_cluster_debug.py tests/test_slice_debug_scripts.py
git commit -m "feat: add cluster debug script"
```

### Task 6: Run full verification and document server usage

**Files:**
- Modify: `tests/test_slice_baseline_cli.py`
- Modify: `docs/plans/2026-03-12-slice-debug-pipeline-design.md`

**Step 1: Add or update verification coverage**

Ensure direct script execution tests cover no-package-context execution for the new debug scripts where practical.

**Step 2: Run full relevant verification**

Run:

```bash
python -m unittest tests.test_processed_feature_assembler tests.test_slice_feature_projector tests.test_slice_finder tests.test_slice_baseline_cli tests.test_slice_debug_scripts -v
```

Expected: PASS

**Step 3: Run smoke verification**

Run a small local end-to-end debug flow on fixture data:

```bash
python run_slice_assembler_debug.py ...
python run_slice_projector_debug.py ...
python run_slice_cluster_debug.py ...
```

Expected: all artifacts written successfully

**Step 4: Commit**

```bash
git add tests/test_slice_baseline_cli.py docs/plans/2026-03-12-slice-debug-pipeline-design.md docs/plans/2026-03-12-slice-debug-pipeline-implementation.md
git commit -m "test: verify slice debug pipeline"
```
