# Remix Recommendation Design

## Goal

Build a first usable `Slice Remix Hub` that can:
- take existing slice artifacts from the `50k` training pool,
- estimate a baseline slice mixture for a fixed training budget,
- generate feasible sparse remix candidates,
- map each candidate to feature-portrait shifts and sample-level execution policy,
- fit a lightweight surrogate from measured experiment outcomes,
- rank candidates and output a structured recommendation object.

The design is intentionally scoped to a **single-budget first version**. The purpose of the first implementation is not to automate the full training pipeline end-to-end, but to establish a correct, testable baseline for offline response-dataset construction and runtime recommendation.

## Top-Level Position in the System

The overall framework now has four top-level modules:

1. `Data Feature Preprocessor`
2. `Slice Workspace`
3. `Slice Remix Hub`
4. `Experiment & Provenance`

`Slice Remix Hub` sits downstream of `Slice Workspace` and consumes:
- sample-level slice memberships,
- slice-level portraits,
- feature-schema-derived portrait groups,
- existing cluster artifacts and projected feature artifacts.

`Experiment & Provenance` stores measured outcomes from training runs and later feeds them back into the surrogate.

## Data Roles

The current `50k` subset is treated as:

- `U_pool`: the remixable training pool

It is **not** treated as a validation/test image set.

Model performance must be measured on an independent evaluation set:

- `E_eval_in`: primary in-domain evaluation set, e.g. `COCO-Stuff val`
- optional `E_eval_ood`: secondary OOD / transfer evaluation set, e.g. `Cityscapes val`

The surrogate is not trained on images directly. It is trained on a **response dataset** whose rows are experiment records:

\[
(p_0, q, \Delta q, \Delta \phi, c, \Delta y)
\]

where:
- \(p_0\): baseline mixture
- \(q\): candidate target mixture
- \(\Delta q = q - p_0\)
- \(\Delta \phi\): feature-level portrait shift
- \(c\): context such as budget and evaluation protocol
- \(\Delta y = y(q) - y(p_0)\): paired gain relative to baseline

## Core Variables

For each sample \(x_i\), slice discovery provides a soft membership:

\[
m_i = (m_{i1}, \dots, m_{iK}), \qquad m_{ik} \ge 0,\quad \sum_{k=1}^K m_{ik}=1
\]

For a baseline subset \(D_0\) of size \(B\):

\[
p_{0,k} = \frac{1}{B}\sum_{x_i \in D_0} m_{ik}
\]

The target recommendation is a slice-level mixture:

\[
q \in \Delta^K,\qquad q_k \ge 0,\quad \sum_{k=1}^K q_k = 1
\]

The user-facing recommendation is:

\[
\Delta q = q - p_0
\]

This is a reallocation problem under fixed training budget, not a total-data-scaling problem.

## Action Space

The feasible remix action space is:

\[
\mathcal{Q}(p_0)
=
\left\{
q = p_0 + \delta
\;\middle|\;
q \in \Delta^K,\;
\sum_k \delta_k = 0,\;
\|\delta\|_0 \le s,\;
\|\delta\|_1 \le \epsilon
\right\}
\]

Interpretation:
- \(\sum_k \delta_k = 0\): budget-preserving reallocation
- \(\|\delta\|_0 \le s\): only a few slices are modified
- \(\|\delta\|_1 \le \epsilon\): remix remains near baseline and practically actionable

The action can also be parameterized as:

\[
q = p_0 + \tau(u-v)
\]

where:
- \(u \in \Delta^K\): receivers distribution
- \(v \in \Delta^K\): donors distribution
- \(\tau\): total transfer amount

`Pairwise transfer` is the special case with one receiver and one donor.

## Feature-Level Portrait Shift

Recommendation should not rely only on slice IDs. It should model how a candidate remix changes the interpretable training-data portrait.

For each processed feature group \(f\), define:

\[
z_i^{(f)} \in \mathbb{R}^{d_f}
\]

as the sample-level processed representation in schema space.

For slice \(k\), define the feature portrait:

\[
s_k^{(f)} = \frac{\sum_i m_{ik} z_i^{(f)}}{\sum_i m_{ik}}
\]

For candidate mixture \(q\):

\[
\phi^{(f)}(q) = \sum_{k=1}^K q_k s_k^{(f)}
\]

and relative to baseline:

\[
\Delta \phi^{(f)} = \phi^{(f)}(q) - \phi^{(f)}(p_0)
\]

The first version will use the eight real processed feature groups already present in the repository:
- `laplacian`
- `noise_pca`
- `bga`
- `small_ratio`
- `visual_semantic_gap`
- `empirical_iou`
- `knn_local_density`
- `prototype_distance`

## Offline Remix Learning

Offline learning constructs the response dataset and fits the surrogate.

### Baseline Trials

Fix a training budget \(B\), e.g. `1000`, and repeatedly sample baseline subsets:

\[
D_0^{(t)} \sim \text{Sample}(U_{pool}, B)
\]

Each trial yields:
- a baseline subset,
- a baseline mixture \(p_0^{(t)}\),
- a baseline model score \(y(p_0^{(t)})\).

### Candidate Design

For each baseline trial, build a small candidate library from portrait-aware sparse actions.

Recommended MVP per baseline trial:
- `8` pairwise candidates
- `2` sparse multi-slice candidates

To avoid selecting only coverage-dominated actions, candidate directions should be diversified in portrait space with group balancing over:
- `quality`
- `difficulty`
- `coverage`

### Supervision Label

For each candidate:

\[
\Delta y = y(q) - y(p_0)
\]

This paired-gain label is preferred over absolute performance because it cancels much of the baseline-specific noise.

## Surrogate Model

The first implementation should use a lightweight, interpretable surrogate.

Recommended main model:

\[
\hat g
=
\beta_q^\top \Delta q
+
\sum_{f=1}^F \beta_f^\top \Delta \phi^{(f)}
+
\Delta q^\top A \Delta q
+
\beta_c^\top c
\]

Interpretation:
- linear term on \(\Delta q\): first-order slice reallocation effects
- linear term on \(\Delta \phi\): feature-distribution effects
- quadratic term on \(\Delta q\): pairwise slice interaction
- context term: budget / protocol / evaluation condition

The first implementation should support:
- a linear baseline model,
- a quadratic-in-action model,
- optional bootstrap uncertainty estimation later.

## Runtime Recommendation

At inference time:
1. estimate current baseline mixture \(p_0\),
2. generate a feasible sparse candidate set \(\mathcal{Q}(p_0)\),
3. compute \(\Delta q\) and \(\Delta \phi\) for each candidate,
4. score each candidate using the trained surrogate,
5. return the best recommendation object.

Recommended runtime score:

\[
\text{Score}(q)
=
\mu_{\hat g}(q)
-
\kappa \sigma_{\hat g}(q)
-
\lambda_1 \|q-p_0\|_1
-
\lambda_2 \|\delta\|_0
\]

The recommendation object should expose:
- `baseline_mixture`
- `target_mixture`
- `delta_q`
- `predicted_gain_mean`
- `predicted_gain_std`
- `risk_adjusted_score`
- `feature_level_rationale`
- `execution_policy_summary`

## Budgeted Execution

Recommendation must ultimately be executed at the sample level.

Practical first implementation:

\[
p_{\text{pool},k} = \frac{1}{N}\sum_{i=1}^N m_{ik}
\]

\[
\rho_k(q) = \frac{q_k}{p_{\text{pool},k}+\varepsilon}
\]

\[
s_i(q) = \sum_{k=1}^K \rho_k(q)m_{ik}
\]

\[
\pi_i(q)=\frac{s_i(q)}{\sum_j s_j(q)}
\]

This provides a sample-level weighting rule derived from the target slice mixture.

More ideal formulation:

\[
\pi^*(q)
=
\arg\min_{\pi \in \Delta^N}
\mathrm{KL}(\pi \,\|\, \pi_0)
\quad
\text{s.t.}
\quad
M^\top \pi = q
\]

but the first implementation will use the simpler weighting approximation.

## Repository Placement

Recommendation code should live in a new package:

- `slice_remix/`

Suggested modules:
- `baseline.py`
- `actions.py`
- `portraits.py`
- `policy.py`
- `dataset.py`
- `surrogate.py`
- `recommender.py`

And two CLI entrypoints:
- `run_remix_response_dataset.py`
- `run_remix_recommendation.py`

This keeps `slice_discovery` focused on discovery/explanation and makes `slice_remix` the dedicated decision layer.

## MVP Scope

The first implementation should build a usable research baseline, not the full production system.

### In Scope
- load existing slice artifacts
- estimate baseline mixture
- generate sparse candidate mixtures
- compute feature-level portrait shifts
- produce sample-level execution weights
- read/write response dataset rows
- fit linear/quadratic surrogate
- score candidate recommendations

### Out of Scope for v1
- automated full segmentation training orchestration
- frontend recommendation UI
- online model updating from live experiments
- multi-budget surrogate conditioning
- active-learning candidate refinement

## Validation Strategy

Surrogate validation should be split by **baseline trials**, not by images.

This tests the true recommendation question:
- given a previously unseen baseline,
- can the surrogate correctly rank or select promising remix candidates?

Important metrics:
- within-trial rank correlation
- top-k hit rate
- regret
- sign accuracy

## Current Next Step

The design is stable enough to start implementation.

The next implementation milestone should be:
1. create `slice_remix/` package,
2. implement artifact loading + baseline estimation,
3. implement sparse candidate generation and portrait shift computation,
4. define response dataset schema and CLI,
5. fit a first surrogate baseline,
6. return a structured recommendation object.
