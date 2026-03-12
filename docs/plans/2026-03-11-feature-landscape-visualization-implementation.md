# Feature Landscape Visualization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a script that reads one sample from the processed quality, difficulty, and coverage bundles and renders a paper-style unified 3D feature landscape plus supporting exports.

**Architecture:** Add a standalone visualization module that aligns processed records by `image_rel`, extracts the eight sequence-like fields defined during design, maps them to a unified `8 x 16` visualization matrix, and exports a 3D landscape figure, a 2D heatmap, and the underlying matrix JSON. Keep the matrix assembly pure and testable; keep plotting isolated so unit tests do not depend on the display backend.

**Tech Stack:** Python 3.10, NumPy, matplotlib, json, argparse, unittest

---

### Task 1: Write the failing tests

**Files:**
- Create: `tests/test_feature_landscape_visualization.py`
- Test: `tests/test_feature_landscape_visualization.py`

**Step 1: Write the failing test**

Add tests that assert:
- short histogram-like sequences are resampled to width 16
- 50-length coverage profiles are pooled to width 16
- a combined matrix assembled from fake processed records has shape `(8, 16)` and expected row labels
- matrix JSON export writes the row labels and grouped metadata

**Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_feature_landscape_visualization -v`
Expected: FAIL because `clip_dinoiser.visualize_feature_landscape` does not exist yet.

### Task 2: Implement the minimal visualization helpers

**Files:**
- Create: `visualize_feature_landscape.py`
- Test: `tests/test_feature_landscape_visualization.py`

**Step 1: Write minimal implementation**

Implement helpers to:
- load processed bundles
- index records by `image_rel`
- resample short sequences to width 16
- pool long profiles to width 16
- assemble the unified `8 x 16` matrix and metadata
- export matrix JSON

**Step 2: Run test to verify it passes**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_feature_landscape_visualization -v`
Expected: PASS

### Task 3: Add plotting and CLI entrypoint

**Files:**
- Modify: `visualize_feature_landscape.py`
- Test: `tests/test_feature_landscape_visualization.py`

**Step 1: Add plotting functions**

Add:
- a 3D paper-style landscape renderer
- a 2D heatmap renderer
- CLI support for `--image-rel`, processed bundle paths, `--output-dir`, `--mode`, and `--save-matrix-json`

**Step 2: Run test to verify non-plot helpers still pass**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_feature_landscape_visualization -v`
Expected: PASS

### Task 4: Generate one sample artifact and verify

**Files:**
- Output: `figures/feature_landscape/*`

**Step 1: Run the visualization script**

Run: `~/.pyenv/versions/3.10.14/bin/python visualize_feature_landscape.py --output-dir ./figures/feature_landscape --mode analysis`
Expected: generates one example 3D PNG, one heatmap PNG, and one matrix JSON using the first shared sample across all three processed bundles.

**Step 2: Run regression tests**

Run: `~/.pyenv/versions/3.10.14/bin/python -m unittest tests.test_feature_landscape_visualization tests.test_processed_feature_postprocessing -v`
Expected: PASS
