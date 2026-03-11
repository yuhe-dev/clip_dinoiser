# Processed Feature Postprocessing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a schema-driven postprocessing pipeline that converts raw quality, difficulty, and coverage feature bundles into processed feature bundles saved under `data/data_feature/<dimension>/`.

**Architecture:** Add one reusable postprocessing module at the repo root that loads the unified schema, computes distribution or profile encodings per feature, writes processed records plus schema/config/summary artifacts, and exposes a CLI for per-dimension or multi-dimension execution. Keep heavy model logic out of the postprocessor; it should only read existing raw `.npy` bundles and config/stats JSON files.

**Tech Stack:** Python 3.10, NumPy, JSON, argparse, unittest

---

### Task 1: Write failing tests for schema-driven encoding helpers

**Files:**
- Create: `tests/test_processed_feature_postprocessing.py`
- Test: `tests/test_processed_feature_postprocessing.py`

**Step 1: Write the failing test**

Add tests that assert:
- a distribution feature is encoded with a normalized histogram and expected summary fields
- a profile feature is encoded with `profile`, `delta_profile`, and expected summary fields
- bundle writers create `*_processed_features.npy`, `*_processed_schema.json`, `*_processing_config.json`, and `*_processed_summary.json`

**Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_processed_feature_postprocessing -v`
Expected: FAIL because `clip_dinoiser.postprocess_feature_bundles` does not exist yet.

### Task 2: Implement the minimal postprocessing module

**Files:**
- Create: `postprocess_feature_bundles.py`
- Test: `tests/test_processed_feature_postprocessing.py`

**Step 1: Write minimal implementation**

Implement helpers to:
- load the unified schema
- fit fixed or robust histogram edges
- encode distribution features
- encode profile features
- process one dimension bundle
- save processed bundle artifacts

**Step 2: Run test to verify it passes**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_processed_feature_postprocessing -v`
Expected: PASS

### Task 3: Run postprocessing for local raw bundles

**Files:**
- Modify: `postprocess_feature_bundles.py`
- Test: `tests/test_processed_feature_postprocessing.py`

**Step 1: Add CLI entrypoint**

Add CLI arguments for:
- `--dimensions`
- `--data-root`
- `--schema-path`
- `--skip-missing`

**Step 2: Run targeted processing**

Run: `~/.pyenv/versions/3.10.14/bin/python postprocess_feature_bundles.py --dimensions difficulty coverage --data-root ./data/data_feature`
Expected: writes processed bundle files for `difficulty` and `coverage`

### Task 4: Verify outputs and regression coverage

**Files:**
- Test: `tests/test_processed_feature_postprocessing.py`
- Output: `data/data_feature/difficulty/*processed*`
- Output: `data/data_feature/coverage/*processed*`

**Step 1: Run tests**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_processed_feature_postprocessing tests.test_quality_raw_feature_scripts tests.test_difficulty_raw_feature_scripts tests.test_coverage_raw_feature_scripts -v`
Expected: PASS

**Step 2: Verify generated artifacts exist**

Run: `find data/data_feature -maxdepth 2 -type f | rg 'processed_(features|schema|summary)|processing_config'`
Expected: processed files for locally available dimensions are listed
