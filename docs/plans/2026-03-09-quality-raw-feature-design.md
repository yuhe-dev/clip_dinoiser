# Quality Raw Feature Design

**Goal:** Refactor the quality feature implementations so `get_vector_score()` returns per-sample raw feature collections instead of final histogram vectors, enabling a later dataset-level postprocessing stage that applies shared global statistics such as min/max before binning.

**Scope:** This change only targets the quality feature classes in `feature_utils/data_feature/implementations/quality.py`: `LaplacianSharpness`, `WeakTexturePCANoise`, and `BoundaryGradientAdherence`. Scalar `get_score()` behavior remains backward-compatible.

## Design

### 1. Interface semantics
- `get_score()` continues to return the legacy scalar score for compatibility.
- `get_vector_score()` is repurposed to return a one-dimensional `np.ndarray(dtype=np.float32)` containing raw feature values for a single sample.
- Returned raw vectors are variable-length by design; later dataset-level code will convert them into fixed-length histogram vectors.

### 2. Raw feature definitions
- `LaplacianSharpness.get_vector_score()` returns one raw Laplacian variance value per sampled patch.
- `WeakTexturePCANoise.get_vector_score()` returns one local noise proxy value per sampled patch, matching the Notion requirement for patch-level noise scores.
- `BoundaryGradientAdherence.get_vector_score()` returns the normalized gradient magnitudes for pixels on the mask boundary, so downstream code can histogram them directly over `[0, 1]`.

### 3. Error handling and edge cases
- All three quality implementations return `np.asarray([], dtype=np.float32)` when no valid raw values can be produced.
- `BoundaryGradientAdherence` still returns an empty array if `mask` is missing or contains no boundary pixels.
- `WeakTexturePCANoise` no longer collapses to a single scalar in `get_vector_score()`; the raw-vector path avoids hidden histogram or reduction logic.

### 4. Testing strategy
- Update Laplacian tests to validate raw-vector semantics instead of histogram shape/normalization.
- Add tests that assert raw vectors are one-dimensional `float32` arrays with the expected value ranges and non-empty behavior on simple synthetic inputs.
- Preserve compatibility tests for scalar `get_score()`.
