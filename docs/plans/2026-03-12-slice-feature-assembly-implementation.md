# Slice Feature Assembly Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the postprocessed-feature assembly layer that produces stable sample-level block and flat representations for downstream soft slice finding.

**Architecture:** Introduce an object-oriented middle layer that loads processed feature bundles, validates alignment by `image_rel`, extracts schema-driven numeric blocks, and saves assembled artifacts as `npz + json`. Keep clustering-specific scaling and weighting out of the assembler itself by following it with a thin projector layer and a small sanity baseline.

**Tech Stack:** Python 3.10, numpy, json, unittest, existing processed feature schema and bundle pipeline

---

### Task 1: Add assembler tests for alignment and block extraction

**Files:**
- Create: `tests/test_processed_feature_assembler.py`
- Reference: `postprocess_feature_bundles.py`
- Reference: `docs/feature_schema/unified_processed_feature_schema.json`

**Step 1: Write the failing test**

```python
def test_assembler_builds_block_and_flat_views_from_processed_records():
    quality_records = [...]
    difficulty_records = [...]
    coverage_records = [...]

    assembler = ProcessedFeatureAssembler.from_processed_records(
        quality_records=quality_records,
        difficulty_records=difficulty_records,
        coverage_records=coverage_records,
        schema=TEST_SCHEMA,
    )

    assert assembler.sample_count == 2
    assert assembler.list_blocks() == [
        "quality.laplacian",
        "difficulty.small_ratio",
        "coverage.knn_local_density",
    ]
    assert assembler.get_block("quality.laplacian").matrix.shape == (2, 4)
    assert assembler.get_flat_view().shape == (2, 11)
```

**Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_processed_feature_assembler -v`
Expected: FAIL with `ImportError` or `NameError` because `ProcessedFeatureAssembler` does not exist yet.

**Step 3: Write minimal implementation**

Create a new assembler module stub with:

```python
class ProcessedFeatureAssembler:
    @classmethod
    def from_processed_records(cls, quality_records, difficulty_records, coverage_records, schema):
        raise NotImplementedError
```

**Step 4: Run test to verify it still fails for the expected missing behavior**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_processed_feature_assembler -v`
Expected: FAIL with `NotImplementedError`.

**Step 5: Commit**

```bash
git add tests/test_processed_feature_assembler.py slice_feature_assembler.py
git commit -m "test: add processed feature assembler skeleton"
```

### Task 2: Implement schema-driven field extraction and block objects

**Files:**
- Create: `slice_feature_assembler.py`
- Test: `tests/test_processed_feature_assembler.py`

**Step 1: Write the failing test**

```python
def test_block_extraction_follows_model_input_field_order():
    block = assembler.get_block("difficulty.small_ratio")
    np.testing.assert_allclose(
        block.matrix[0],
        np.asarray([0.2, 0.3, 1.0, 0.0, 0.5], dtype=np.float32),
    )
```

**Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_processed_feature_assembler.ProcessedFeatureAssemblerTests.test_block_extraction_follows_model_input_field_order -v`
Expected: FAIL because block extraction order is not implemented.

**Step 3: Write minimal implementation**

Implement:

```python
@dataclass
class FeatureBlock:
    name: str
    dimension: str
    feature_name: str
    field_names: list[str]
    matrix: np.ndarray
    normalization_hints: dict[str, str]
```

Add extraction helpers that:
- read `model_input_fields`
- expand scalar fields to length 1
- preserve `hist`, `profile`, `delta_profile` lengths
- concatenate in declared order
- cast to `np.float32`

**Step 4: Run test to verify it passes**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_processed_feature_assembler -v`
Expected: PASS for field-order and matrix-shape tests.

**Step 5: Commit**

```bash
git add tests/test_processed_feature_assembler.py slice_feature_assembler.py
git commit -m "feat: implement schema-driven feature block extraction"
```

### Task 3: Implement alignment validation and metadata generation

**Files:**
- Modify: `slice_feature_assembler.py`
- Test: `tests/test_processed_feature_assembler.py`

**Step 1: Write the failing test**

```python
def test_alignment_validation_rejects_mismatched_image_ids():
    with self.assertRaises(ValueError):
        ProcessedFeatureAssembler.from_processed_records(
            quality_records=[{"image_rel": "a.jpg", ...}],
            difficulty_records=[{"image_rel": "b.jpg", ...}],
            coverage_records=[{"image_rel": "a.jpg", ...}],
            schema=TEST_SCHEMA,
        )
```

**Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_processed_feature_assembler.ProcessedFeatureAssemblerTests.test_alignment_validation_rejects_mismatched_image_ids -v`
Expected: FAIL because alignment validation is missing.

**Step 3: Write minimal implementation**

Add validation that:
- enforces identical ordered `image_rel` lists across dimensions
- rejects duplicates
- stores `sample_ids`
- builds metadata with block order, per-block column ranges, and source schema versions

**Step 4: Run test to verify it passes**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_processed_feature_assembler -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_processed_feature_assembler.py slice_feature_assembler.py
git commit -m "feat: validate bundle alignment and expose assembler metadata"
```

### Task 4: Add processed-bundle loading and artifact persistence

**Files:**
- Modify: `slice_feature_assembler.py`
- Test: `tests/test_processed_feature_assembler.py`

**Step 1: Write the failing test**

```python
def test_save_and_load_round_trip_preserves_flat_and_block_views():
    assembler.save(tmpdir)
    restored = ProcessedFeatureAssembler.load(tmpdir)

    np.testing.assert_allclose(restored.get_flat_view(), assembler.get_flat_view())
    self.assertEqual(restored.list_blocks(), assembler.list_blocks())
```

**Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_processed_feature_assembler.ProcessedFeatureAssemblerTests.test_save_and_load_round_trip_preserves_flat_and_block_views -v`
Expected: FAIL because persistence is not implemented.

**Step 3: Write minimal implementation**

Implement:
- `from_processed_paths(...)`
- `save(output_dir)` writing:
  - `assembled_features.npz`
  - `assembled_features_meta.json`
- `load(output_dir)`

Save:
- `sample_ids`
- `X_flat`
- one array per block
- metadata for block order, field names, column ranges, source paths

**Step 4: Run test to verify it passes**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_processed_feature_assembler -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_processed_feature_assembler.py slice_feature_assembler.py
git commit -m "feat: persist assembled feature artifacts"
```

### Task 5: Add projector tests for field-aware scaling and block weighting

**Files:**
- Create: `tests/test_slice_feature_projector.py`
- Create: `slice_feature_projector.py`
- Reference: `slice_feature_assembler.py`

**Step 1: Write the failing test**

```python
def test_projector_scales_scalar_fields_but_preserves_distribution_vectors():
    projector = SliceFeatureProjector(
        scalar_scaler="zscore",
        block_weighting="equal_by_block",
        pca_components=None,
    )
    result = projector.fit_transform(assembler)

    assert result.matrix.shape[0] == assembler.sample_count
    assert result.block_ranges["quality.laplacian"] == (0, 4)
```

**Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_slice_feature_projector -v`
Expected: FAIL because projector does not exist yet.

**Step 3: Write minimal implementation**

Create a projector class that:
- selects blocks
- identifies scalar vs vector fields from block metadata
- z-scores scalar columns only
- leaves hist/profile/delta_profile columns untouched
- applies optional equal-by-block weighting

**Step 4: Run test to verify it passes**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_slice_feature_projector -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_slice_feature_projector.py slice_feature_projector.py
git commit -m "feat: add slice feature projector baseline"
```

### Task 6: Add optional PCA support and projector persistence

**Files:**
- Modify: `slice_feature_projector.py`
- Modify: `tests/test_slice_feature_projector.py`

**Step 1: Write the failing test**

```python
def test_projector_can_emit_pca_view_and_save_state():
    projector = SliceFeatureProjector(pca_components=3)
    projected = projector.fit_transform(assembler)
    projector.save(tmpdir)
    restored = SliceFeatureProjector.load(tmpdir)

    assert projected.matrix.shape == (assembler.sample_count, 3)
```

**Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_slice_feature_projector.SliceFeatureProjectorTests.test_projector_can_emit_pca_view_and_save_state -v`
Expected: FAIL because PCA and persistence are missing.

**Step 3: Write minimal implementation**

Add:
- optional PCA projection
- persisted config and fitted parameters
- returned metadata describing input block ranges and output dimensionality

**Step 4: Run test to verify it passes**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_slice_feature_projector -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_slice_feature_projector.py slice_feature_projector.py
git commit -m "feat: add projector PCA and persistence"
```

### Task 7: Add soft k-means sanity baseline

**Files:**
- Create: `slice_finder.py`
- Create: `tests/test_slice_finder.py`
- Reference: `slice_feature_projector.py`

**Step 1: Write the failing test**

```python
def test_soft_kmeans_returns_membership_rows_that_sum_to_one():
    finder = SoftKMeansSliceFinder(num_slices=3, seed=0)
    result = finder.fit(projected_matrix, sample_ids=["a", "b", "c"])

    np.testing.assert_allclose(result.membership.sum(axis=1), np.ones(3))
```

**Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_slice_finder -v`
Expected: FAIL because finder and result objects do not exist.

**Step 3: Write minimal implementation**

Implement:
- `SoftKMeansSliceFinder`
- `SliceFindingResult`
- membership normalization
- hard assignment
- per-slice mass

**Step 4: Run test to verify it passes**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_slice_finder -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_slice_finder.py slice_finder.py
git commit -m "feat: add soft k-means slice finding baseline"
```

### Task 8: Add GMM baseline and slice summary helpers

**Files:**
- Modify: `slice_finder.py`
- Modify: `tests/test_slice_finder.py`

**Step 1: Write the failing test**

```python
def test_gmm_slice_finder_emits_slice_profiles_and_representative_indices():
    finder = GMMSliceFinder(num_slices=3, covariance_type="diag", seed=0)
    result = finder.fit(projected_matrix, sample_ids=sample_ids)

    assert len(result.slice_ids) == 3
    assert "slice_00" in result.slice_statistics
```

**Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_slice_finder.SliceFinderTests.test_gmm_slice_finder_emits_slice_profiles_and_representative_indices -v`
Expected: FAIL because GMM baseline and slice summaries are missing.

**Step 3: Write minimal implementation**

Implement:
- `GMMSliceFinder`
- posterior membership output
- slice-level statistics:
  - slice mass
  - center
  - representative sample indices
  - top shifted flat dimensions

**Step 4: Run test to verify it passes**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_slice_finder -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_slice_finder.py slice_finder.py
git commit -m "feat: add GMM slice finding baseline"
```

### Task 9: Add a CLI script that runs the first end-to-end baseline

**Files:**
- Create: `run_slice_finding_baseline.py`
- Modify: `tests/test_slice_finder.py`
- Reference: `slice_feature_assembler.py`
- Reference: `slice_feature_projector.py`
- Reference: `slice_finder.py`

**Step 1: Write the failing test**

```python
def test_cli_writes_membership_and_metadata_artifacts():
    exit_code = main([
        "--data-root", tmp_data_root,
        "--output-dir", tmp_output_dir,
        "--finder", "soft_kmeans",
        "--num-slices", "4",
    ])
    assert exit_code == 0
    assert os.path.exists(os.path.join(tmp_output_dir, "slice_result.npz"))
```

**Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_slice_finder -v`
Expected: FAIL because CLI entrypoint is missing.

**Step 3: Write minimal implementation**

Implement a CLI that:
- loads processed bundles from `data/data_feature`
- assembles features
- projects them
- runs soft k-means or GMM
- writes result artifacts and a summary json

**Step 4: Run test to verify it passes**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_processed_feature_assembler tests.test_slice_feature_projector tests.test_slice_finder -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add run_slice_finding_baseline.py tests/test_processed_feature_assembler.py tests/test_slice_feature_projector.py tests/test_slice_finder.py slice_feature_assembler.py slice_feature_projector.py slice_finder.py
git commit -m "feat: add end-to-end slice finding baseline pipeline"
```

### Task 10: Verify the real 50k pipeline and document usage

**Files:**
- Modify: `README.md`
- Create: `docs/plans/2026-03-12-slice-feature-assembly-implementation.md`

**Step 1: Write the failing verification checklist**

```text
- assembler loads all 50000 processed samples
- projector emits expected matrix shapes
- soft k-means and GMM both finish and write artifacts
- output metadata records block ranges and source bundle paths
```

**Step 2: Run real verification**

Run: `~/.pyenv/versions/3.10.14/bin/python run_slice_finding_baseline.py --data-root ./data/data_feature --output-dir ./artifacts/slice_baseline --finder soft_kmeans --num-slices 8`

Run: `~/.pyenv/versions/3.10.14/bin/python run_slice_finding_baseline.py --data-root ./data/data_feature --output-dir ./artifacts/slice_baseline_gmm --finder gmm --num-slices 8 --pca-components 64`

Expected: both commands exit successfully and write `npz + json` artifacts.

**Step 3: Write minimal documentation**

Add README usage notes with:

```bash
~/.pyenv/versions/3.10.14/bin/python run_slice_finding_baseline.py \
  --data-root ./data/data_feature \
  --output-dir ./artifacts/slice_baseline \
  --finder gmm \
  --num-slices 8 \
  --block-weighting equal_by_block \
  --pca-components 64
```

**Step 4: Run verification tests**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_processed_feature_assembler tests.test_slice_feature_projector tests.test_slice_finder -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add README.md run_slice_finding_baseline.py tests/test_processed_feature_assembler.py tests/test_slice_feature_projector.py tests/test_slice_finder.py slice_feature_assembler.py slice_feature_projector.py slice_finder.py
git commit -m "docs: document slice feature assembly baseline workflow"
```
