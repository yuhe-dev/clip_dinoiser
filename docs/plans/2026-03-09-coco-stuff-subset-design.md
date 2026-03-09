# COCO-Stuff 50k Subset Design

**Goal:** Create a reusable 50,000-sample public experiment subset under `data/` that preserves the original `coco_stuff_164k` directory structure and serves as the shared asset for feature computation, slice finding, remix weighting, and downstream evaluations.

**Architecture:** Add a dataset-preparation script that discovers the train split of the source COCO-Stuff dataset, samples 50,000 image/annotation pairs with `seed=0`, and materializes a lightweight subset directory containing index metadata plus mirrored relative paths for images and annotations. The subset is treated as a train-only experimental corpus and becomes the canonical input for later feature precomputation classes.

**Data Layout:** The new subset lives under `data/` with stable paths, a manifest/index file, a config file recording sampling parameters, and copied/symlinked references that mirror the original image and annotation tree so later pipelines can reuse existing path logic.

**Execution Notes:** The script prints progress while scanning, sampling, and writing outputs, and emits example saved records at the end for verification.
