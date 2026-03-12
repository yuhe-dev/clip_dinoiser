# Feature Pipeline Refactor Design

## Goal

Refactor the current feature-engineering code for `quality`, `difficulty`, and `coverage` into reusable modules while preserving the existing external data contract:

- the same raw bundle file organization
- the same processed bundle file organization
- the same raw and processed field names
- the same schema-driven postprocess behavior

The refactor is code-organization work only. It must not modify or regenerate the existing feature files that are already being used by downstream slice-finding development.

## Scope

This refactor covers:

- raw feature extraction orchestration
- raw bundle assembly, stats, and I/O
- processed feature postprocessing orchestration
- a unified pipeline runner that can chain raw extraction and postprocess

This refactor does not cover:

- changing the feature semantics
- changing existing output filenames or field names
- changing existing schema JSON files
- adding visualization or downstream analysis into the pipeline

## Current Pain Points

The current codebase already has good single-feature implementation classes under `feature_utils/data_feature/implementations/`, but the higher-level logic is fragmented:

- each extraction script mixes dataset traversal, feature instantiation, record assembly, stats, config, and saving
- raw bundle logic is duplicated across `quality`, `difficulty`, and `coverage`
- postprocess logic is centralized in one script but still behaves like a script, not a reusable module
- there is no clean object model for `raw extraction -> raw bundle -> postprocess -> processed bundle`
- there is no single pipeline entry point that can run the full feature pipeline while still preserving current outputs

## Design Principles

1. External compatibility first

The current raw and processed bundle structure is treated as a fixed contract. Internal refactoring must preserve:

- current raw feature keys such as `laplacian_raw`, `small_ratio_raw`, `small_ratio_num_values`
- current processed feature block structure
- current filenames such as `quality_raw_features.npy` and `difficulty_processed_features.npy`

2. Single-feature logic stays in `implementations`

Classes in `feature_utils/data_feature/implementations/` should continue to own only per-feature computation logic. They should not absorb dataset traversal, bundle writing, or postprocess orchestration.

3. Schema-driven postprocess remains centralized

Processed feature encoding should continue to be driven by the unified processed schema rather than spreading encoding rules back into multiple scripts.

4. Gradual migration

Existing scripts stay in place initially, but they are reduced to thin wrappers over the new modular classes. This avoids disrupting current slice-finding work.

## Architectural Layers

The refactored design has four layers plus a thin compatibility shell.

### 1. Feature Implementations

Location:

- `feature_utils/data_feature/implementations/quality.py`
- `feature_utils/data_feature/implementations/difficulty.py`
- `feature_utils/data_feature/implementations/coverage.py`

Responsibility:

- compute a single feature for a single sample
- return raw values or profile values

Examples:

- `LaplacianSharpness`
- `WeakTexturePCANoise`
- `BoundaryGradientAdherence`
- `SmallObjectRatioCOCOStuff`
- `SemanticAmbiguityCLIP`
- `EmpiricalDifficultyMaskClip`
- `KNNLocalDensityCLIPFaiss`
- `PrototypeMarginCLIPFaiss`

These classes remain the feature-computation layer only.

### 2. Raw Extraction Layer

New package:

- `feature_utils/data_feature/extraction/`

Responsibility:

- traverse dataset samples
- instantiate the required feature implementations for one dimension
- load sample context such as image, mask, or embedding lookup path
- align outputs to the raw record field schema

Key classes:

- `BaseRawExtractor`
- `QualityRawExtractor`
- `DifficultyRawExtractor`
- `CoverageRawExtractor`

The extractor layer answers:

"Given a dataset sample, which feature classes should be called, and how should their outputs be mapped into raw record fields?"

### 3. Bundle Layer

New package:

- `feature_utils/data_feature/bundle/`

Responsibility:

- represent raw and processed bundles as explicit objects
- compute stats and summaries
- load and save bundles using the existing on-disk format

Key classes:

- `RawFeatureBundle`
- `ProcessedFeatureBundle`
- `BundleStatsBuilder`
- `RawBundleIO`
- `ProcessedBundleIO`

The bundle layer answers:

"How should a set of records, stats, config, and schema metadata be organized and persisted?"

### 4. Postprocess Layer

New package:

- `feature_utils/data_feature/postprocess/`

Responsibility:

- load the unified processed schema
- resolve dimension-specific schema
- fit any distribution encoders
- transform raw records into processed feature records

Key classes:

- `DistributionFeatureEncoder`
- `ProfileFeatureEncoder`
- `FeaturePostprocessor`
- `SchemaResolver`

The postprocess layer answers:

"How should a raw bundle be aligned to the standard processed schema while preserving feature semantics?"

### 5. Pipeline Layer

New package:

- `feature_utils/data_feature/pipeline/`

Responsibility:

- instantiate extractors and postprocessors
- run raw extraction only, postprocess only, or the full pipeline
- provide one place for orchestration without redefining lower-level logic

Key classes:

- `PipelineConfig`
- `FeaturePipelineFactory`
- `DataFeaturePipelineRunner`

The pipeline layer answers:

"How should extraction, raw bundling, and postprocessing be instantiated and chained together?"

## Proposed Directory Layout

```text
feature_utils/data_feature/
  implementations/
    quality.py
    difficulty.py
    coverage.py

  extraction/
    __init__.py
    base.py
    quality.py
    difficulty.py
    coverage.py

  bundle/
    __init__.py
    raw_bundle.py
    processed_bundle.py
    stats.py
    io.py

  postprocess/
    __init__.py
    encoders.py
    processor.py
    schema.py

  pipeline/
    __init__.py
    config.py
    factory.py
    runner.py
```

## Class Responsibilities and Interfaces

### Extraction Layer

#### `BaseRawExtractor`

This class provides common extraction flow:

- progress and logging
- feature-instance construction
- per-sample context loading
- per-record assembly

Recommended interface:

```python
class BaseRawExtractor:
    dimension_name: str

    def extract_records(
        self,
        subset_root: str,
        subset_records: Sequence[dict],
        feature_meta: dict,
        show_progress: bool = True,
        progress_interval: int = 100,
    ) -> list[dict]:
        ...

    def build_feature_instances(self, feature_meta: dict) -> dict:
        ...

    def load_sample_context(self, subset_root: str, record: dict) -> dict:
        ...

    def extract_single_record(
        self,
        record: dict,
        sample_context: dict,
        feature_instances: dict,
        feature_meta: dict,
    ) -> dict:
        ...
```

#### `QualityRawExtractor`

Maps:

- `laplacian_raw`
- `noise_pca_raw`
- `bga_raw`

#### `DifficultyRawExtractor`

Maps:

- `small_ratio_raw`
- `small_ratio_num_values`
- `visual_semantic_gap_raw`
- `empirical_iou_raw`

#### `CoverageRawExtractor`

Maps:

- `knn_neighbor_distances_raw`
- `prototype_distances_raw`

### Bundle Layer

#### `RawFeatureBundle`

Fields:

- `dimension_name`
- `records`
- `stats`
- `feature_config`

#### `ProcessedFeatureBundle`

Fields:

- `dimension_name`
- `records`
- `schema`
- `processing_config`
- `summary`

#### `BundleStatsBuilder`

Responsibilities:

- raw feature global stats
- processed feature summary

#### `RawBundleIO`

Responsibilities:

- save using the current raw bundle filenames
- load existing raw bundles

#### `ProcessedBundleIO`

Responsibilities:

- save using the current processed bundle filenames
- load existing processed bundles

### Postprocess Layer

#### `DistributionFeatureEncoder`

Responsibilities:

- value transform
- fit global bin edges
- normalized histogram encoding
- summary generation

#### `ProfileFeatureEncoder`

Responsibilities:

- profile and delta-profile generation
- support count propagation via `source_count_key`
- summary generation

#### `FeaturePostprocessor`

Recommended interface:

```python
class FeaturePostprocessor:
    def process_bundle(
        self,
        raw_bundle: RawFeatureBundle,
        dimension_schema: dict,
        progress_interval: int = 100,
    ) -> ProcessedFeatureBundle:
        ...
```

#### `SchemaResolver`

Responsibilities:

- load unified processed schema
- return the schema block for a specific dimension

### Pipeline Layer

#### `FeaturePipelineFactory`

Responsibilities:

- choose the right extractor for a dimension
- create postprocessor instances

#### `DataFeaturePipelineRunner`

Recommended interface:

```python
class DataFeaturePipelineRunner:
    def run_raw(...):
        ...

    def run_postprocess(...):
        ...

    def run_full(...):
        ...
```

`run_full(...)` is the unified path from dataset input to processed bundle output.

## `raw extraction` vs `raw bundle`

These are two different concepts and should remain different abstractions.

### Raw Extraction

Raw extraction is the computation stage.

It is responsible for:

- reading sample inputs
- calling feature implementations
- producing in-memory raw feature records

Example output for one sample:

```python
{
  "image_rel": "...",
  "annotation_rel": "...",
  "laplacian_raw": ...,
  "noise_pca_raw": ...,
  "bga_raw": ...
}
```

### Raw Bundle

Raw bundle is the organization-and-persistence stage.

It is responsible for:

- storing a set of raw records
- computing global stats
- storing feature config metadata
- saving the records, stats, and config using the existing bundle layout

Example files:

- `quality_raw_features.npy`
- `quality_global_stats.json`
- `quality_feature_config.json`

In short:

- raw extraction = produce raw feature values
- raw bundle = package and save those values

## Migration Strategy

### Step 1: Extract reusable modules

Add the new extraction, bundle, postprocess, and pipeline modules while preserving all current output contracts.

Current scripts stay available:

- `extract_quality_raw_features.py`
- `extract_difficulty_raw_features.py`
- `extract_coverage_raw_features.py`
- `postprocess_feature_bundles.py`

But internally they become orchestration wrappers over the new module classes.

### Step 2: Add unified pipeline entry point

Add a new script:

- `run_feature_pipeline.py`

This script supports:

- raw extraction only
- postprocess only
- full pipeline

without changing the output format.

### Step 3: Keep old scripts as compatibility wrappers

Once the new runner is stable, the old scripts remain as thin wrappers for compatibility and convenience.

## Unified Pipeline CLI

Recommended CLI shape:

```bash
python run_feature_pipeline.py \
  --dimensions quality difficulty coverage \
  --stage full \
  --subset-root ./data/coco_stuff50k \
  --data-root ./data/data_feature \
  --schema-path ./docs/feature_schema/unified_processed_feature_schema.json
```

Recommended arguments:

- `--dimensions`
- `--stage raw|postprocess|full`
- `--subset-root`
- `--index-path`
- `--data-root`
- `--schema-path`
- `--embedding-root`
- `--dataset-module-path`
- `--progress-interval`
- `--skip-missing`

Dimension-specific feature-meta options should still be supported, such as:

- `--patch-size`
- `--stride`
- `--clip-model`
- `--clip-pretrained`
- `--knn-k`
- `--prototype-top-m`

## Testing Strategy

Because this is a refactor, the most important requirement is compatibility.

Testing should cover four areas:

### 1. Extractor tests

Verify:

- expected raw field names are emitted
- `small_ratio_num_values` is preserved
- missing-input behavior matches the current scripts

### 2. Bundle I/O tests

Verify:

- raw bundle filenames remain unchanged
- processed bundle filenames remain unchanged
- save/load cycles preserve records and metadata

### 3. Postprocess compatibility tests

Verify:

- processed feature block structure stays unchanged
- `source_count_key` behavior stays correct
- distribution and profile encoders preserve current behavior

### 4. Pipeline integration tests

Verify:

- `run_raw`
- `run_postprocess`
- `run_full`

all produce the correct bundle structure on toy data.

## Recommendation

Proceed with a compatibility-first modular refactor:

- keep single-feature computation in `implementations`
- move dimension-level orchestration into extractor classes
- move bundle structure and persistence into explicit bundle objects
- move schema-driven normalization into reusable postprocess classes
- add a unified runner without changing existing output contracts

This achieves cleaner code organization, stronger reuse, easier extension, and a stable path toward a single end-to-end feature pipeline entry point while keeping current downstream work untouched.
