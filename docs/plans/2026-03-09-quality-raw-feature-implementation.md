# Quality Raw Feature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the quality feature implementations so `get_vector_score()` returns raw per-sample feature collections for Laplacian, local noise, and boundary gradients while preserving legacy scalar `get_score()` behavior.

**Architecture:** Update `feature_utils/data_feature/implementations/quality.py` so each quality metric exposes a raw one-dimensional `float32` array from `get_vector_score()`. Adapt the tests to validate raw-vector semantics and keep scalar compatibility checks where they already exist.

**Tech Stack:** Python, NumPy, OpenCV stubs in tests, `unittest`

---

### Task 1: Update Laplacian tests for raw vectors

**Files:**
- Modify: `tests/test_laplacian_vectorization.py`
- Test: `tests/test_laplacian_vectorization.py`

**Step 1: Write the failing test**

```python
def test_vector_score_returns_patch_raw_scores(self):
    metric = LaplacianSharpness()
    values = metric.get_vector_score(self._make_image(96, 96), meta={"patch_size": 32, "stride": 16})
    self.assertEqual(values.dtype, np.float32)
    self.assertEqual(values.ndim, 1)
    self.assertEqual(values.shape[0], 25)
```

**Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_laplacian_vectorization -v`
Expected: FAIL because `get_vector_score()` still returns histogram bins.

**Step 3: Write minimal implementation**

Update `LaplacianSharpness.get_vector_score()` in `feature_utils/data_feature/implementations/quality.py` to return raw per-patch Laplacian variance values.

**Step 4: Run test to verify it passes**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_laplacian_vectorization -v`
Expected: PASS

### Task 2: Add raw-vector tests for noise and BGA

**Files:**
- Create: `tests/test_quality_raw_vector_scores.py`
- Modify: `feature_utils/data_feature/implementations/quality.py`
- Test: `tests/test_quality_raw_vector_scores.py`

**Step 1: Write the failing test**

```python
def test_noise_pca_vector_score_returns_patchwise_raw_values(self):
    ...

def test_bga_vector_score_returns_boundary_gradient_values(self):
    ...
```

**Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_quality_raw_vector_scores -v`
Expected: FAIL because the implementations still return scalar-wrapped vectors.

**Step 3: Write minimal implementation**

Refactor `WeakTexturePCANoise.get_vector_score()` to compute local patch noise proxies and refactor `BoundaryGradientAdherence.get_vector_score()` to return normalized boundary gradient magnitudes as a one-dimensional array.

**Step 4: Run test to verify it passes**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_quality_raw_vector_scores -v`
Expected: PASS

### Task 3: Run focused regression tests

**Files:**
- Modify: `feature_utils/data_feature/implementations/quality.py`
- Test: `tests/test_laplacian_vectorization.py`
- Test: `tests/test_quality_raw_vector_scores.py`
- Test: `tests/test_metric_vector_contract.py`

**Step 1: Run the focused suite**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_laplacian_vectorization tests.test_quality_raw_vector_scores tests.test_metric_vector_contract -v`
Expected: PASS

**Step 2: If needed, do minimal cleanup**

Tighten helper naming or dtype conversions only if the tests expose duplication or brittle behavior.

**Step 3: Re-run the focused suite**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_laplacian_vectorization tests.test_quality_raw_vector_scores tests.test_metric_vector_contract -v`
Expected: PASS
