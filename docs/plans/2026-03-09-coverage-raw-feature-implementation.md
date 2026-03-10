# Coverage Raw Feature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a complete 50k-specific coverage pipeline that regenerates visual embedding assets, computes prototype centroids, and extracts raw coverage features for neighbor distances and prototype distance profiles.

**Architecture:** Refactor the coverage feature classes to return raw one-dimensional arrays from `get_vector_score()`, then add three supporting scripts: one for CLIP embedding generation, one for prototype generation, and one for batch raw-feature extraction plus sanity summarization. All outputs live under `data/data_feature/coverage/` and are aligned to `data/coco_stuff50k/sample_index.npy`.

**Tech Stack:** Python, NumPy, FAISS, OpenCLIP, unittest

---

### Task 1: Add coverage vectorization tests

**Files:**
- Create: `tests/test_coverage_vectorization.py`
- Modify: `feature_utils/data_feature/implementations/coverage.py`
- Modify: `feature_utils/data_feature/registry.py`

**Step 1: Write the failing test**

```python
def test_knn_vector_score_returns_neighbor_distances():
    ...

def test_prototype_vector_score_returns_top_m_profile():
    ...
```

**Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_coverage_vectorization -v`
Expected: FAIL because both coverage classes still return scalar-wrapped vectors.

**Step 3: Write minimal implementation**

Update `feature_utils/data_feature/implementations/coverage.py` so `get_vector_score()` exposes raw neighbor distances and prototype profiles while keeping `get_score()` backward-compatible.

**Step 4: Run test to verify it passes**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_coverage_vectorization -v`
Expected: PASS

### Task 2: Add coverage asset helper tests

**Files:**
- Create: `tests/test_coverage_raw_feature_scripts.py`
- Create: `extract_coverage_embeddings.py`
- Create: `build_coverage_prototypes.py`
- Create: `extract_coverage_raw_features.py`
- Create: `sanity_check_coverage_raw_features.py`

**Step 1: Write the failing test**

```python
def test_embedding_metadata_writer_saves_paths_and_config():
    ...

def test_prototype_writer_saves_centroids_and_config():
    ...

def test_coverage_summary_reports_lengths_and_ranges():
    ...
```

**Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_coverage_raw_feature_scripts -v`
Expected: FAIL because the scripts and helper functions do not exist yet.

**Step 3: Write minimal implementation**

Implement helper functions for embedding metadata saving, prototype saving, coverage raw bundle saving, and coverage sanity summaries.

**Step 4: Run test to verify it passes**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_coverage_raw_feature_scripts -v`
Expected: PASS

### Task 3: Add embedding generation script

**Files:**
- Create: `extract_coverage_embeddings.py`
- Modify: `tests/test_coverage_raw_feature_scripts.py`

**Step 1: Write the failing test**

```python
def test_limit_subset_records_truncates_embedding_input():
    ...
```

**Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_coverage_raw_feature_scripts -v`
Expected: FAIL because record limiting and embedding config helpers are missing.

**Step 3: Write minimal implementation**

Implement the embedding extraction CLI using `coco_stuff50k/sample_index.npy`, OpenCLIP image encoding, and stable path alignment output files.

**Step 4: Run test to verify it passes**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_coverage_raw_feature_scripts -v`
Expected: PASS

### Task 4: Add prototype build and coverage extraction scripts

**Files:**
- Create: `build_coverage_prototypes.py`
- Create: `extract_coverage_raw_features.py`
- Create: `sanity_check_coverage_raw_features.py`
- Modify: `tests/test_coverage_raw_feature_scripts.py`

**Step 1: Write the failing test**

```python
def test_compute_coverage_global_stats_aggregates_knn_and_prototype_values():
    ...
```

**Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_coverage_raw_feature_scripts -v`
Expected: FAIL because the coverage raw extraction helpers do not aggregate or summarize these features yet.

**Step 3: Write minimal implementation**

Implement prototype saving, coverage raw extraction, and summary generation using the new 50k embedding assets.

**Step 4: Run test to verify it passes**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_coverage_raw_feature_scripts -v`
Expected: PASS

### Task 5: Run focused regression suite

**Files:**
- Test: `tests/test_coverage_vectorization.py`
- Test: `tests/test_coverage_raw_feature_scripts.py`
- Test: `tests/test_difficulty_vectorization.py`
- Test: `tests/test_quality_raw_feature_scripts.py`
- Test: `tests/test_metric_vector_contract.py`

**Step 1: Run the focused suite**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_coverage_vectorization tests.test_coverage_raw_feature_scripts tests.test_difficulty_vectorization tests.test_quality_raw_feature_scripts tests.test_metric_vector_contract -v`
Expected: PASS

**Step 2: If needed, do minimal cleanup**

Only remove duplication or tighten helper names if the test outputs reveal brittle behavior.

**Step 3: Re-run the focused suite**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_coverage_vectorization tests.test_coverage_raw_feature_scripts tests.test_difficulty_vectorization tests.test_quality_raw_feature_scripts tests.test_metric_vector_contract -v`
Expected: PASS
