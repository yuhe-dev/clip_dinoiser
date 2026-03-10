# Coverage Raw Feature Design

**Goal:** Build a full coverage-feature pipeline for the `coco_stuff50k` subset, including fresh visual-embedding assets, prototype generation, and raw coverage feature extraction aligned with the Notion definitions for neighbor-distance histograms and prototype distance profiles.

## Scope
This work replaces the previous 20k-only coverage assets with a 50k-specific pipeline rooted at `data/coco_stuff50k/sample_index.npy`. It covers asset generation and raw feature extraction for the two coverage sub-features:
- KNN-based Local Density
- Prototype Centroid Distance

## Design

### 1. Asset pipeline
Coverage requires fresh embedding-space assets for the 50k subset:
- Generate CLIP visual embeddings for all subset images.
- Save embeddings and aligned image paths under a stable coverage-specific directory.
- Generate prototype centroids from the 50k embeddings and save them alongside the embedding assets.

### 2. Raw feature semantics
- `KNNLocalDensityCLIPFaiss.get_vector_score()` returns the distances to the top-k nearest neighbors as a raw one-dimensional array.
- `PrototypeMarginCLIPFaiss.get_vector_score()` returns the sorted top-m prototype distances or the derived top-m profile needed for later fixed-length encoding.
- `get_score()` stays backward-compatible for scalar use cases.

### 3. Batch extraction
A dedicated extraction script reads the 50k subset index, uses the fresh embedding/prototype assets, and saves per-sample raw coverage features plus global summary statistics. A paired sanity script summarizes lengths, value ranges, and sample previews.

### 4. Data layout
Suggested output layout:
- `data/data_feature/coverage/visual_embedding/visual_emb.npy`
- `data/data_feature/coverage/visual_embedding/clip_paths_abs.json`
- `data/data_feature/coverage/visual_embedding/prototypes_k200.npy`
- `data/data_feature/coverage/coverage_raw_features.npy`
- `data/data_feature/coverage/coverage_global_stats.json`
- `data/data_feature/coverage/coverage_feature_config.json`
- `data/data_feature/coverage/coverage_raw_summary.json`

### 5. Testing strategy
- Add unit tests for raw vector semantics of the two coverage feature classes.
- Add unit tests for embedding/prototype metadata helpers and coverage bundle summary helpers.
- Keep heavy model execution out of unit tests by testing helper functions and small synthetic arrays.
