# Unified Processed Feature Schema

This document defines the final post-processing schema for `quality`, `difficulty`, and `coverage` after raw feature extraction and before slice finding.

It is paired with [unified_processed_feature_schema.json](/Users/yuhe/research/UIST%202026/clip_dinoiser/docs/feature_schema/unified_processed_feature_schema.json), which is the machine-readable source of truth.

For a concrete fake-data example showing exactly what one processed sample record looks like after all three dimensions are encoded, see [example_processed_feature_record.json](/Users/yuhe/research/UIST%202026/clip_dinoiser/docs/feature_schema/example_processed_feature_record.json).

## Design Principles

1. Raw feature extraction and dataset-level post-processing stay separate.
2. Histogram bins are fit once at the dataset level, not per image.
3. Variable-length raw arrays are converted into aligned per-sample representations.
4. Already aligned rank profiles are preserved as profiles instead of being re-histogrammed.
5. Every feature block keeps both distribution or profile shape and a small amount of support information.

## Why `num_values` and `log_num_values` both exist

- `num_values` is the literal count of raw values observed for the feature in one image.
- `log_num_values` is `log1p(num_values)` and is the version intended for numerical downstream input.

Examples:

- `laplacian_raw`: number of patch-level sharpness values
- `noise_pca_raw`: number of patch-level noise values
- `bga_raw`: number of boundary-pixel gradient values
- `small_ratio_raw`: number of valid connected components used for the profile
- `visual_semantic_gap_raw`: number of valid regions
- `empirical_iou_raw`: number of valid class or region IoUs
- `knn_neighbor_distances_raw`: number of returned neighbors, typically `k`
- `prototype_distances_raw`: number of returned prototype distances, typically `top_m`

The count itself is useful for debugging and interpretation. The log-compressed version is safer for model input because counts can vary by orders of magnitude.

## Common Block Semantics

Every processed feature block should keep these common fields:

- `encoding`
- `empty_flag`
- `num_values`
- `log_num_values`

### Stored Fields vs `model_input_fields`

There are always two views of the processed data:

- Stored fields
  The full processed feature block saved to disk. This includes the main representation (`hist` or `profile`), support fields, and summary fields.
- `model_input_fields`
  The recommended subset of stored fields that should be concatenated into the final numeric vector for slice finding.

This separation is deliberate:

- it keeps the processed bundle interpretable
- it avoids losing information too early
- it lets downstream experiments change feature selection without recomputing processed records

So `model_input_fields` is not the whole stored schema. It is the default downstream selection from the stored schema.

### `empty_flag`

- `0`: valid values exist
- `1`: the raw value list is empty, so the main representation is filled with zeros

This is important because an all-zero histogram or profile can mean either "truly no evidence" or "valid values all happen to land elsewhere". The explicit flag removes that ambiguity.

### Distribution Block Template

Distribution-encoded features such as `laplacian`, `noise_pca`, `bga`, `visual_semantic_gap`, and `empirical_iou` should be stored like this:

```json
{
  "encoding": "distribution",
  "value_transform": "log1p or identity",
  "empty_flag": 0,
  "num_values": 128,
  "log_num_values": 4.8598,
  "hist": [0.0, 0.1, 0.2, 0.3],
  "summary": {
    "mean": 0.0,
    "std": 0.0,
    "q10": 0.0,
    "q50": 0.0,
    "q90": 0.0
  },
  "model_input_fields": ["hist", "log_num_values", "empty_flag"]
}
```

### Profile Block Template

Profile-encoded features such as `small_ratio`, `knn_local_density`, and `prototype_distance` should be stored like this:

```json
{
  "encoding": "profile",
  "value_transform": "identity",
  "empty_flag": 0,
  "num_values": 16,
  "log_num_values": 2.8332,
  "profile": [0.1, 0.2, 0.4, 0.6],
  "delta_profile": [0.1, 0.1, 0.2, 0.2],
  "summary": {
    "mean": 0.0,
    "std": 0.0
  },
  "model_input_fields": ["delta_profile", "log_num_values", "empty_flag"]
}
```

## Quality

### Goal

`quality` should describe the internal quality distribution of one image, not just a single score.

### `laplacian`

- Raw source: `laplacian_raw`
- Meaning: patch-level local sharpness
- Encoding: distribution
- Transform: `log1p`
- Histogram: 12 bins
- Bin fitting: global robust range over transformed values using `p1-p99`

Rationale:

- The raw values are positive and long-tailed.
- A log transform improves stability without destroying the ordering semantics.
- A normalized histogram preserves the "shape" of local sharpness within one image.

Important derived fields:

- `low_sharpness_mass`: concentration in the low-sharpness bins
- `high_sharpness_mass`: concentration in the high-sharpness bins

Recommended downstream use:

- `hist`
- `log_num_values`
- `empty_flag`
- `q50`
- `q90`
- `low_sharpness_mass`

### `noise_pca`

- Raw source: `noise_pca_raw`
- Meaning: patch-level local noise proxy
- Encoding: distribution
- Transform: `log1p`
- Histogram: 12 bins
- Bin fitting: global robust range over transformed values using `p1-p99`

Rationale:

- It is also positive and heavy-tailed.
- The important semantics are not just average noise, but how much of the image is clean versus noisy.

Important derived fields:

- `low_noise_mass`
- `high_noise_mass`

Recommended downstream use:

- `hist`
- `log_num_values`
- `empty_flag`
- `q50`
- `q90`
- `high_noise_mass`

### `bga`

- Raw source: `bga_raw`
- Meaning: boundary-gradient alignment values in `[0, 1]`
- Encoding: distribution
- Transform: identity
- Histogram: 8 bins
- Bin fitting: fixed range `[0, 1]`

Rationale:

- The raw scale already has direct semantics.
- Using fixed bins preserves interpretability across runs and datasets.

Important derived fields:

- `low_bga_mass`: mass in poorly aligned boundary bins
- `high_bga_mass`: mass in strongly aligned boundary bins

Recommended downstream use:

- `hist`
- `log_num_values`
- `empty_flag`
- `q10`
- `q50`
- `low_bga_mass`
- `high_bga_mass`

## Difficulty

### Goal

`difficulty` is heterogeneous. Some features are already meaningful fixed profiles, while others are variable-length event sets that should become distributions.

### `small_ratio`

- Raw source: `small_ratio_raw`
- Source count field: `small_ratio_num_values`
- Meaning: cumulative ratio curve of small-object prevalence across thresholds
- Encoding: profile
- Transform: identity
- Length: 16

Why this is not histogrammed again:

- The current raw output is already a semantically aligned threshold profile.
- Re-binning it would destroy the threshold semantics.

Derived fields:

- `delta_profile`: converts the cumulative profile into per-interval mass
- `first_active_bin`
- `mass_small_extreme`
- `mass_small_mid`

Field details:

- `num_values`
  Number of valid connected components or object regions that contributed to the profile. In processed bundles this should come from the raw field `small_ratio_num_values`, not from the profile length.
- `log_num_values`
  `log1p(num_values)`. This is the numerically safer version for downstream input.
- `profile`
  The original cumulative threshold curve from the raw feature.
- `delta_profile`
  The interval-mass version of `profile`, where `delta[0] = profile[0]` and `delta[i] = max(profile[i] - profile[i - 1], 0)`.
- `first_active_bin`
  The first non-zero `delta_profile` position, normalized to `[0, 1]`. Earlier means extremely small objects appear earlier in the threshold schedule.
- `mass_small_extreme`
  The sum of the earliest delta bins, representing the concentration of very tiny objects.
- `mass_small_mid`
  The sum of middle delta bins, representing the concentration of moderate small-object mass.

Recommended downstream use:

- `delta_profile`
- `log_num_values`
- `empty_flag`
- `mass_small_extreme`
- `mass_small_mid`

### `visual_semantic_gap`

- Raw source: `visual_semantic_gap_raw`
- Meaning: per-region ambiguity or mismatch in visual-text alignment
- Encoding: distribution
- Transform: `log1p`
- Histogram: 12 bins
- Bin fitting: global robust range over transformed values using `p1-p99`

Derived fields:

- `high_gap_mass`

Recommended downstream use:

- `hist`
- `log_num_values`
- `empty_flag`
- `q50`
- `q90`
- `high_gap_mass`

### `empirical_iou`

- Raw source: `empirical_iou_raw`
- Meaning: per-class or per-region empirical IoU values
- Encoding: distribution
- Transform: identity
- Histogram: 8 bins
- Bin fitting: fixed range `[0, 1]`

Derived fields:

- `low_iou_mass`
- `high_iou_mass`

Recommended downstream use:

- `hist`
- `log_num_values`
- `empty_flag`
- `q10`
- `q50`
- `low_iou_mass`
- `high_iou_mass`

## Coverage

### Goal

`coverage` describes where a sample lives in embedding space. The raw outputs are already rank-aligned vectors with stable positional meaning, so the final encoding should preserve that geometry.

### `knn_local_density`

- Raw source: `knn_neighbor_distances_raw`
- Meaning: sorted distances to the nearest `k` neighbors in CLIP space
- Encoding: profile
- Transform: identity with non-negative clamp
- Length: `feature_meta.knn_k`, currently 50

Why this stays a profile:

- Position in the vector already means rank among nearest neighbors.
- The distance curve itself is the semantic object.
- Re-histogramming would throw away whether the density drop happens immediately or gradually.

Derived fields:

- `delta_profile`: gap between successive neighbor distances
- `nearest_distance`
- `farthest_distance`
- `density_score = 1 / (eps + mean)`

Recommended downstream use:

- `profile`
- `q10`
- `q50`
- `q90`
- `nearest_distance`
- `density_score`

### `prototype_distance`

- Raw source: `prototype_distances_raw`
- Meaning: sorted distances to the closest embedding prototypes
- Encoding: profile
- Transform: identity with non-negative clamp
- Length: `feature_meta.prototype_top_m`, currently 50

Why this stays a profile:

- Rank order to prototypes is already aligned.
- The early part of the curve captures how tightly a sample belongs to one prototype versus how diffusely it relates to many.

Derived fields:

- `delta_profile`
- `nearest_prototype_distance`
- `prototype_margin_top2`
- `prototype_margin_top5`

Recommended downstream use:

- `profile`
- `q10`
- `q50`
- `q90`
- `nearest_prototype_distance`
- `prototype_margin_top2`
- `prototype_margin_top5`

## Empty-Value Policy

### For distribution-encoded features

If the raw array is empty:

- `hist` is all zeros
- all summary values are zero
- `empty_flag = 1`
- `num_values = 0`
- `log_num_values = 0`

### For profile-encoded features

If the raw array is empty:

- `profile` is all zeros with the configured length
- `delta_profile` is all zeros with the configured length
- all summary values are zero
- `empty_flag = 1`
- `num_values = 0`
- `log_num_values = 0`

## Recommended Bundle Layout

Recommended processed artifacts per dimension:

- `data/data_feature/<dimension>/<dimension>_processed_features.npy`
- `data/data_feature/<dimension>/<dimension>_processed_schema.json`
- `data/data_feature/<dimension>/<dimension>_processing_config.json`
- `data/data_feature/<dimension>/<dimension>_processed_summary.json`

The machine-readable schema in this directory should be copied or referenced by each dimension-specific processed bundle so downstream scripts can always recover field order and semantics.
