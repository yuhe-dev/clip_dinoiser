# Feature Pipeline Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor raw extraction, raw bundle assembly, and processed postprocessing into reusable modules while preserving the existing raw/processed file layout and field schema.

**Architecture:** Keep single-feature computation inside `feature_utils/data_feature/implementations/`, add extractor classes for dimension-level raw orchestration, add explicit raw/processed bundle objects plus reusable I/O and stats builders, and add schema-driven postprocess classes. Existing scripts remain as compatibility wrappers, and a new unified runner provides one end-to-end entry point without changing current output contracts.

**Tech Stack:** Python 3.9+, NumPy, argparse, unittest

---

### Task 1: Add bundle data models and raw stats builder

**Files:**
- Create: `feature_utils/data_feature/bundle/__init__.py`
- Create: `feature_utils/data_feature/bundle/raw_bundle.py`
- Create: `feature_utils/data_feature/bundle/processed_bundle.py`
- Create: `feature_utils/data_feature/bundle/stats.py`
- Test: `tests/test_feature_bundle_models.py`

**Step 1: Write the failing test**

```python
def test_raw_bundle_stats_builder_matches_existing_global_stats_shape():
    records = [
        {"laplacian_raw": np.asarray([1.0, 2.0], dtype=np.float32)},
        {"laplacian_raw": np.asarray([], dtype=np.float32)},
    ]
    stats = build_raw_feature_stats(records, feature_keys=("laplacian_raw",))
    assert stats["features"]["laplacian_raw"]["total_values"] == 2
    assert stats["features"]["laplacian_raw"]["empty_samples"] == 1
```

```python
def test_processed_bundle_summary_matches_existing_processed_summary_shape():
    processed_records = [
        {"features": {"laplacian": {"empty_flag": 0, "num_values": 3}}},
        {"features": {"laplacian": {"empty_flag": 1, "num_values": 0}}},
    ]
    summary = build_processed_feature_summary(processed_records)
    assert summary["features"]["laplacian"]["empty_samples"] == 1
```

**Step 2: Run test to verify it fails**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_feature_bundle_models -v
```

Expected: FAIL because the new bundle/stats module does not exist yet.

**Step 3: Write minimal implementation**

Create:

- `RawFeatureBundle`
- `ProcessedFeatureBundle`
- `build_raw_feature_stats(...)`
- `build_processed_feature_summary(...)`

Keep stat keys compatible with the existing JSON layout.

**Step 4: Run test to verify it passes**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_feature_bundle_models -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_feature_bundle_models.py feature_utils/data_feature/bundle
git commit -m "feat(bundle): add feature bundle models and stats helpers"
```

### Task 2: Add bundle I/O classes compatible with current filenames

**Files:**
- Create: `feature_utils/data_feature/bundle/io.py`
- Modify: `feature_utils/data_feature/bundle/__init__.py`
- Test: `tests/test_feature_bundle_io.py`

**Step 1: Write the failing test**

```python
def test_raw_bundle_io_uses_existing_quality_filenames():
    bundle = RawFeatureBundle(
        dimension_name="quality",
        records=[{"image_rel": "a.jpg"}],
        stats={"num_samples": 1, "features": {}},
        feature_config={"subset_root": "data/coco_stuff50k"},
    )
    paths = RawBundleIO().save(bundle, tmpdir)
    assert paths["records_path"].endswith("quality_raw_features.npy")
    assert paths["stats_path"].endswith("quality_global_stats.json")
    assert paths["config_path"].endswith("quality_feature_config.json")
```

```python
def test_processed_bundle_io_uses_existing_difficulty_filenames():
    bundle = ProcessedFeatureBundle(
        dimension_name="difficulty",
        records=[{"image_rel": "a.jpg", "features": {}}],
        schema={"schema_version": "difficulty.v1"},
        processing_config={},
        summary={"num_samples": 1, "features": {}},
    )
    paths = ProcessedBundleIO().save(bundle, tmpdir)
    assert paths["records_path"].endswith("difficulty_processed_features.npy")
```

**Step 2: Run test to verify it fails**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_feature_bundle_io -v
```

Expected: FAIL because `RawBundleIO` and `ProcessedBundleIO` do not exist yet.

**Step 3: Write minimal implementation**

Implement:

- `RawBundleIO.save(...)`
- `RawBundleIO.load(...)`
- `ProcessedBundleIO.save(...)`
- `ProcessedBundleIO.load(...)`

Use the current bundle filename contract exactly.

**Step 4: Run test to verify it passes**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_feature_bundle_io -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_feature_bundle_io.py feature_utils/data_feature/bundle/io.py feature_utils/data_feature/bundle/__init__.py
git commit -m "feat(bundle): add raw and processed bundle io"
```

### Task 3: Add extractor base class and quality extractor

**Files:**
- Create: `feature_utils/data_feature/extraction/__init__.py`
- Create: `feature_utils/data_feature/extraction/base.py`
- Create: `feature_utils/data_feature/extraction/quality.py`
- Test: `tests/test_quality_raw_extractor.py`

**Step 1: Write the failing test**

```python
def test_quality_raw_extractor_emits_existing_raw_field_names():
    extractor = QualityRawExtractor(
        feature_factory=lambda meta: {
            "laplacian": StubFeature([1.0, 2.0]),
            "noise_pca": StubFeature([3.0]),
            "bga": StubFeature([0.5]),
        }
    )
    records = extractor.extract_records(...)
    assert "laplacian_raw" in records[0]
    assert "noise_pca_raw" in records[0]
    assert "bga_raw" in records[0]
```

**Step 2: Run test to verify it fails**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_quality_raw_extractor -v
```

Expected: FAIL because extractor classes do not exist yet.

**Step 3: Write minimal implementation**

Implement:

- `BaseRawExtractor`
- `QualityRawExtractor`

Allow dependency injection for test stubs instead of hard-coding OpenCV and the real feature classes in tests.

**Step 4: Run test to verify it passes**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_quality_raw_extractor -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_quality_raw_extractor.py feature_utils/data_feature/extraction
git commit -m "feat(extraction): add base and quality raw extractors"
```

### Task 4: Add difficulty and coverage extractors

**Files:**
- Create: `feature_utils/data_feature/extraction/difficulty.py`
- Create: `feature_utils/data_feature/extraction/coverage.py`
- Test: `tests/test_difficulty_raw_extractor.py`
- Test: `tests/test_coverage_raw_extractor.py`

**Step 1: Write the failing test**

```python
def test_difficulty_raw_extractor_preserves_small_ratio_num_values():
    extractor = DifficultyRawExtractor(...)
    records = extractor.extract_records(...)
    assert records[0]["small_ratio_num_values"] == 3
```

```python
def test_coverage_raw_extractor_emits_existing_raw_field_names():
    extractor = CoverageRawExtractor(...)
    records = extractor.extract_records(...)
    assert "knn_neighbor_distances_raw" in records[0]
    assert "prototype_distances_raw" in records[0]
```

**Step 2: Run test to verify it fails**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_difficulty_raw_extractor tests.test_coverage_raw_extractor -v
```

Expected: FAIL because the extractors do not exist yet.

**Step 3: Write minimal implementation**

Implement:

- `DifficultyRawExtractor`
- `CoverageRawExtractor`

Preserve existing field names exactly.

**Step 4: Run test to verify it passes**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_difficulty_raw_extractor tests.test_coverage_raw_extractor -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_difficulty_raw_extractor.py tests/test_coverage_raw_extractor.py feature_utils/data_feature/extraction/difficulty.py feature_utils/data_feature/extraction/coverage.py
git commit -m "feat(extraction): add difficulty and coverage raw extractors"
```

### Task 5: Add schema resolver and reusable postprocess encoders

**Files:**
- Create: `feature_utils/data_feature/postprocess/__init__.py`
- Create: `feature_utils/data_feature/postprocess/schema.py`
- Create: `feature_utils/data_feature/postprocess/encoders.py`
- Test: `tests/test_feature_postprocess_encoders.py`

**Step 1: Write the failing test**

```python
def test_distribution_encoder_matches_existing_hist_encoding():
    encoder = DistributionFeatureEncoder(spec)
    encoder.fit([np.asarray([0.5, 1.5, 2.5], dtype=np.float32)])
    encoded = encoder.transform(np.asarray([0.5, 1.5, 2.5], dtype=np.float32), {})
    np.testing.assert_allclose(encoded["hist"].sum(), 1.0)
```

```python
def test_profile_encoder_uses_source_count_key_when_present():
    encoder = ProfileFeatureEncoder(spec={"source_count_key": "small_ratio_num_values", ...})
    encoded = encoder.transform(np.asarray([0.0, 0.5, 1.0], dtype=np.float32), {"small_ratio_num_values": 7})
    assert encoded["num_values"] == 7
```

**Step 2: Run test to verify it fails**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_feature_postprocess_encoders -v
```

Expected: FAIL because the reusable encoder classes do not exist yet.

**Step 3: Write minimal implementation**

Implement:

- `SchemaResolver`
- `DistributionFeatureEncoder`
- `ProfileFeatureEncoder`

Move logic out of `postprocess_feature_bundles.py` without changing behavior.

**Step 4: Run test to verify it passes**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_feature_postprocess_encoders -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_feature_postprocess_encoders.py feature_utils/data_feature/postprocess
git commit -m "feat(postprocess): add reusable schema resolver and encoders"
```

### Task 6: Add dimension-level postprocessor returning processed bundles

**Files:**
- Create: `feature_utils/data_feature/postprocess/processor.py`
- Modify: `feature_utils/data_feature/postprocess/__init__.py`
- Test: `tests/test_feature_postprocessor.py`

**Step 1: Write the failing test**

```python
def test_feature_postprocessor_returns_processed_bundle_with_existing_record_shape():
    raw_bundle = RawFeatureBundle(...)
    bundle = FeaturePostprocessor(schema_resolver).process_bundle(raw_bundle, dimension_schema)
    assert bundle.dimension_name == "difficulty"
    assert "small_ratio" in bundle.records[0]["features"]
```

**Step 2: Run test to verify it fails**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_feature_postprocessor -v
```

Expected: FAIL because `FeaturePostprocessor` does not exist yet.

**Step 3: Write minimal implementation**

Implement `FeaturePostprocessor.process_bundle(...)` so it:

- fits distribution encoders first
- transforms all records
- builds processed summary
- returns `ProcessedFeatureBundle`

**Step 4: Run test to verify it passes**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_feature_postprocessor -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_feature_postprocessor.py feature_utils/data_feature/postprocess/processor.py feature_utils/data_feature/postprocess/__init__.py
git commit -m "feat(postprocess): add feature postprocessor"
```

### Task 7: Convert existing extraction scripts into thin wrappers over the new modules

**Files:**
- Modify: `extract_quality_raw_features.py`
- Modify: `extract_difficulty_raw_features.py`
- Modify: `extract_coverage_raw_features.py`
- Test: `tests/test_quality_raw_feature_scripts.py`
- Test: `tests/test_difficulty_raw_feature_scripts.py`
- Test: `tests/test_coverage_raw_feature_scripts.py`

**Step 1: Write the failing test**

Add or extend tests so they call the public script helper functions and assert:

- output field names remain unchanged
- stats and config remain unchanged in structure
- `small_ratio_num_values` remains present for difficulty

**Step 2: Run test to verify it fails**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_quality_raw_feature_scripts tests.test_difficulty_raw_feature_scripts tests.test_coverage_raw_feature_scripts -v
```

Expected: FAIL because the scripts still implement their own logic instead of using the new modules.

**Step 3: Write minimal implementation**

Refactor the three scripts so they:

- parse CLI args
- load subset records / class names
- instantiate the new extractor class
- build `RawFeatureBundle`
- save via `RawBundleIO`

Preserve current output filenames and printed status messages.

**Step 4: Run test to verify it passes**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_quality_raw_feature_scripts tests.test_difficulty_raw_feature_scripts tests.test_coverage_raw_feature_scripts -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add extract_quality_raw_features.py extract_difficulty_raw_features.py extract_coverage_raw_features.py tests/test_quality_raw_feature_scripts.py tests/test_difficulty_raw_feature_scripts.py tests/test_coverage_raw_feature_scripts.py
git commit -m "refactor(extraction): route raw extraction scripts through extractor modules"
```

### Task 8: Convert `postprocess_feature_bundles.py` into a thin wrapper over the new postprocess modules

**Files:**
- Modify: `postprocess_feature_bundles.py`
- Test: `tests/test_processed_feature_postprocessing.py`

**Step 1: Write the failing test**

Extend the existing tests so they exercise the public dimension bundle processing path and assert that:

- processed records still match the current structure
- summary and config files still use the current names
- `source_count_key` still controls `small_ratio.num_values`

**Step 2: Run test to verify it fails**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_processed_feature_postprocessing -v
```

Expected: FAIL because the script still directly owns the encoding logic.

**Step 3: Write minimal implementation**

Refactor `postprocess_feature_bundles.py` so it becomes:

- schema loading
- raw bundle loading
- postprocessor invocation
- processed bundle save

while preserving CLI behavior and output names.

**Step 4: Run test to verify it passes**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_processed_feature_postprocessing -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add postprocess_feature_bundles.py tests/test_processed_feature_postprocessing.py
git commit -m "refactor(postprocess): route script through reusable postprocess modules"
```

### Task 9: Add unified pipeline config, factory, and runner

**Files:**
- Create: `feature_utils/data_feature/pipeline/__init__.py`
- Create: `feature_utils/data_feature/pipeline/config.py`
- Create: `feature_utils/data_feature/pipeline/factory.py`
- Create: `feature_utils/data_feature/pipeline/runner.py`
- Test: `tests/test_feature_pipeline_runner.py`

**Step 1: Write the failing test**

```python
def test_pipeline_runner_can_run_raw_stage_with_existing_bundle_outputs():
    runner = DataFeaturePipelineRunner(factory=stub_factory)
    result = runner.run_raw(...)
    assert result["records_path"].endswith("quality_raw_features.npy")
```

```python
def test_pipeline_runner_can_run_full_stage():
    runner = DataFeaturePipelineRunner(factory=stub_factory)
    result = runner.run_full(...)
    assert result["processed"]["records_path"].endswith("quality_processed_features.npy")
```

**Step 2: Run test to verify it fails**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_feature_pipeline_runner -v
```

Expected: FAIL because the unified runner does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- `PipelineConfig`
- `FeaturePipelineFactory`
- `DataFeaturePipelineRunner`

Support:

- raw only
- postprocess only
- full pipeline

**Step 4: Run test to verify it passes**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_feature_pipeline_runner -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_feature_pipeline_runner.py feature_utils/data_feature/pipeline
git commit -m "feat(pipeline): add unified feature pipeline runner"
```

### Task 10: Add new unified CLI entry point without removing old script entry points

**Files:**
- Create: `run_feature_pipeline.py`
- Test: `tests/test_run_feature_pipeline_cli.py`

**Step 1: Write the failing test**

```python
def test_run_feature_pipeline_cli_accepts_stage_and_dimensions():
    parser = build_argparser()
    args = parser.parse_args(["--dimensions", "quality", "difficulty", "--stage", "full"])
    assert args.stage == "full"
    assert args.dimensions == ["quality", "difficulty"]
```

**Step 2: Run test to verify it fails**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_run_feature_pipeline_cli -v
```

Expected: FAIL because the new CLI script does not exist yet.

**Step 3: Write minimal implementation**

Implement `run_feature_pipeline.py` as a CLI wrapper over `DataFeaturePipelineRunner`.

Support:

- `--dimensions`
- `--stage raw|postprocess|full`
- `--subset-root`
- `--index-path`
- `--data-root`
- `--schema-path`
- `--embedding-root`
- `--progress-interval`
- `--skip-missing`

and pass through the current dimension-specific feature-meta args.

**Step 4: Run test to verify it passes**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_run_feature_pipeline_cli -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add run_feature_pipeline.py tests/test_run_feature_pipeline_cli.py
git commit -m "feat(cli): add unified feature pipeline entry point"
```

### Task 11: Run compatibility regression suite

**Files:**
- Modify: none unless regressions are found
- Test:
  - `tests/test_feature_bundle_models.py`
  - `tests/test_feature_bundle_io.py`
  - `tests/test_quality_raw_extractor.py`
  - `tests/test_difficulty_raw_extractor.py`
  - `tests/test_coverage_raw_extractor.py`
  - `tests/test_feature_postprocess_encoders.py`
  - `tests/test_feature_postprocessor.py`
  - `tests/test_feature_pipeline_runner.py`
  - `tests/test_run_feature_pipeline_cli.py`
  - `tests/test_quality_raw_feature_scripts.py`
  - `tests/test_difficulty_raw_feature_scripts.py`
  - `tests/test_coverage_raw_feature_scripts.py`
  - `tests/test_processed_feature_postprocessing.py`

**Step 1: Run the full regression suite**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python -m unittest \
  tests.test_feature_bundle_models \
  tests.test_feature_bundle_io \
  tests.test_quality_raw_extractor \
  tests.test_difficulty_raw_extractor \
  tests.test_coverage_raw_extractor \
  tests.test_feature_postprocess_encoders \
  tests.test_feature_postprocessor \
  tests.test_feature_pipeline_runner \
  tests.test_run_feature_pipeline_cli \
  tests.test_quality_raw_feature_scripts \
  tests.test_difficulty_raw_feature_scripts \
  tests.test_coverage_raw_feature_scripts \
  tests.test_processed_feature_postprocessing -v
```

Expected: PASS

**Step 2: Manually sanity-check CLI help**

Run:

```bash
~/.pyenv/versions/3.10.14/bin/python run_feature_pipeline.py --help
~/.pyenv/versions/3.10.14/bin/python extract_quality_raw_features.py --help
~/.pyenv/versions/3.10.14/bin/python postprocess_feature_bundles.py --help
```

Expected: all commands print valid help and preserve expected CLI options.

**Step 3: Commit**

```bash
git add .
git commit -m "test(pipeline): verify refactor compatibility end to end"
```
