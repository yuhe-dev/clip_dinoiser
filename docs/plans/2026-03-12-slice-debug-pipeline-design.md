# Slice Debug Pipeline Design

**Date:** 2026-03-12

**Goal:** Add a layered debug path for slice discovery so assembler, projector, and clustering can be run and validated independently on the server.

## Problem

The current baseline entrypoint runs the entire slice discovery pipeline in one process:

1. load processed bundles
2. assemble sample-level features
3. project the features
4. run clustering
5. save final slice results

This is convenient for a healthy path, but it is a poor debugging surface for server-only failures such as segmentation faults. When the pipeline crashes, there is no durable evidence showing which layer completed successfully, what the intermediate matrix shapes were, or whether invalid values appeared before clustering.

## Proposed Approach

Introduce three direct-run debug scripts, one for each layer:

1. `run_slice_assembler_debug.py`
2. `run_slice_projector_debug.py`
3. `run_slice_cluster_debug.py`

Each script should:

- run exactly one layer
- save a stable intermediate artifact for the next layer
- print a concise stage summary to stdout
- write a JSON metadata/debug report for later inspection

This keeps the failure boundary explicit. If assembler succeeds and projector succeeds but clustering crashes, the user still has persisted intermediate outputs and can inspect them offline.

## Layer Outputs

### Assembler

Inputs:

- processed quality bundle
- processed difficulty bundle
- processed coverage bundle
- unified processed schema

Outputs:

- `assembled_features.npz`
- `assembled_features_meta.json`
- `assembler_debug.json`

Debug report should include:

- sample count
- block order
- each block shape
- flat matrix shape
- per-block finite check
- per-block min/max/mean

### Projector

Inputs:

- assembler artifact directory

Outputs:

- `projected_features.npz`
- `projected_features_meta.json`
- `projector_debug.json`

Debug report should include:

- projected matrix shape
- block ranges
- overall finite check
- per-block min/max/mean/std after projection
- projector config

### Cluster

Inputs:

- projector artifact directory

Outputs:

- `slice_result.npz`
- `slice_result_meta.json`
- `cluster_debug.json`

Debug report should include:

- finder type
- input matrix shape
- whether matrix is finite before fit
- final membership shape
- row-sum min/max
- slice weights
- hard assignment counts
- iterative diagnostics when available

For GMM specifically, the debug path should also persist per-iteration log-likelihood values so a failure or divergence can be localized to a specific iteration range.

## CLI Design

All three scripts should support direct file execution with `python script.py ...` and should not require module-mode invocation. Paths should be explicit and should default to the existing repository layout where reasonable.

The scripts should also support a small smoke-slice of the data for server debugging:

- `--limit-samples N` at assembler level

This allows users to confirm the staged pipeline on a manageable subset before scaling to 50k samples.

## Data Format

Artifacts should remain `npz + json`.

Reasoning:

- already consistent with the current pipeline
- easy to inspect from Python
- stable across scripts
- avoids pickling custom classes into debug artifacts

## Verification Strategy

Tests should cover:

- assembler debug script writes the expected artifacts
- projector debug script loads assembler outputs and writes projected outputs
- cluster debug script loads projector outputs and writes final/debug outputs
- direct script execution works without package context

Manual verification on the server should proceed layer-by-layer:

1. run assembler debug
2. inspect block stats and finite checks
3. run projector debug
4. inspect projected stats
5. run cluster debug with `soft_kmeans`
6. run cluster debug with `gmm`

This order isolates whether the crash is data loading, projection, or clustering specific.
