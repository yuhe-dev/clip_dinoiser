# COCO-Stuff Subset Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reusable script that samples 50,000 COCO-Stuff train pairs with `seed=0`, materializes a subset under `data/`, and records manifest/config metadata for later feature computation.

**Architecture:** Implement a standalone Python tool under `tools/` that scans `images/train2017` and `annotations/train2017`, pairs files by stem, randomly samples a deterministic subset, and writes a mirrored subset directory plus machine-readable metadata. Keep the logic decomposed into pure helper functions so unit tests can validate pairing, sampling, and manifest writing without requiring the full dataset locally.

**Tech Stack:** Python standard library, `numpy`, `unittest`

---

### Task 1: Add test coverage for subset sampling helpers

**Files:**
- Create: `tests/test_coco_stuff_subset_sampling.py`
- Modify: `tools/sample_coco_stuff_subset.py`

**Step 1: Write the failing test**

```python
def test_collect_pairs_matches_images_with_labeltrainids():
    ...

def test_sample_pairs_is_deterministic():
    ...

def test_write_subset_manifest_preserves_relative_paths():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_coco_stuff_subset_sampling -v`
Expected: FAIL with missing module or missing functions.

**Step 3: Write minimal implementation**

Implement pure helpers for pair discovery, deterministic sampling, and manifest/config writing.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_coco_stuff_subset_sampling -v`
Expected: PASS

### Task 2: Implement CLI subset materialization script

**Files:**
- Modify: `tools/sample_coco_stuff_subset.py`
- Test: `tests/test_coco_stuff_subset_sampling.py`

**Step 1: Write the failing test**

```python
def test_main_creates_subset_tree_and_logs_examples():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_coco_stuff_subset_sampling -v`
Expected: FAIL because CLI helpers do not create files/logs yet.

**Step 3: Write minimal implementation**

Implement directory creation, copy/symlink mode, progress logging, subset index/config writing, and example output.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_coco_stuff_subset_sampling -v`
Expected: PASS
