# Small Ratio Count Backfill Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add real object-count support for `small_ratio`, store it in raw difficulty bundles, and propagate it into processed difficulty features.

**Architecture:** Extend `SmallObjectRatioCOCOStuff` with a helper that returns both the cumulative small-ratio profile and the number of valid connected components. Update difficulty raw extraction to save the count field. Update postprocessing to prefer a schema-declared source count key over the profile length fallback. Add a focused backfill script that scans annotation masks and patches an existing `difficulty_raw_features.npy` in place or to a new output path.

**Tech Stack:** Python 3.10, NumPy, argparse, unittest

---

### Task 1: Write failing tests

**Files:**
- Modify: `tests/test_difficulty_vectorization.py`
- Modify: `tests/test_difficulty_raw_feature_scripts.py`
- Modify: `tests/test_processed_feature_postprocessing.py`

**Step 1: Write the failing test**

Add tests that assert:
- `SmallObjectRatioCOCOStuff` can return both profile and connected-component count
- difficulty raw bundle persistence includes `small_ratio_num_values`
- profile postprocessing uses the source count field instead of profile length when available

**Step 2: Run tests to verify they fail**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_difficulty_vectorization tests.test_difficulty_raw_feature_scripts tests.test_processed_feature_postprocessing -v`
Expected: FAIL because no count helper or source-count handling exists yet.

### Task 2: Implement count propagation

**Files:**
- Modify: `feature_utils/data_feature/implementations/difficulty.py`
- Modify: `extract_difficulty_raw_features.py`
- Modify: `postprocess_feature_bundles.py`
- Modify: `docs/feature_schema/unified_processed_feature_schema.json`
- Modify: `docs/feature_schema/unified_processed_feature_schema.md`

**Step 1: Write minimal implementation**

Implement:
- `SmallObjectRatioCOCOStuff.get_profile_and_count(...)`
- raw extraction field `small_ratio_num_values`
- optional schema field `source_count_key`
- postprocessing support that reads the count field and stores the real `num_values`

**Step 2: Run tests to verify they pass**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_difficulty_vectorization tests.test_difficulty_raw_feature_scripts tests.test_processed_feature_postprocessing -v`
Expected: PASS

### Task 3: Add backfill utility

**Files:**
- Create: `backfill_small_ratio_counts.py`

**Step 1: Implement the backfill script**

The script should:
- read `difficulty_raw_features.npy`
- locate annotation masks under a provided `--subset-root`
- compute `small_ratio_num_values` for each record
- write an updated raw bundle and refreshed stats

**Step 2: Verify script help and dry wiring**

Run: `~/.pyenv/versions/3.10.14/bin/python backfill_small_ratio_counts.py --help`
Expected: usage output with required paths
