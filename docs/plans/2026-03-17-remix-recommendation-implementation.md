# Remix Recommendation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a first end-to-end remix recommendation baseline that loads slice artifacts, constructs sparse remix candidates, computes feature-portrait shifts, fits a lightweight surrogate from experiment rows, and ranks runtime recommendations.

**Architecture:** Create a new `slice_remix` package rather than extending `slice_discovery`. The package will reuse projected/cluster artifacts as source-of-truth inputs, define a stable response-dataset schema, and provide two CLI entrypoints: one for response-dataset generation and one for runtime recommendation. The first iteration will use practical sample-weight execution and a linear/quadratic surrogate, leaving full training orchestration and frontend integration out of scope.

**Tech Stack:** Python 3, NumPy, standard library JSON/dataclasses, existing `slice_discovery` artifacts, `unittest`

---

### Task 1: Create the package skeleton

**Files:**
- Create: `slice_remix/__init__.py`
- Create: `slice_remix/types.py`
- Test: `tests/test_slice_remix_types.py`

**Step 1: Write the failing test**

```python
from slice_remix.types import RemixContext, RecommendationResult


def test_recommendation_result_can_be_constructed():
    result = RecommendationResult(
        baseline_mixture=[0.5, 0.5],
        target_mixture=[0.6, 0.4],
        delta_q=[0.1, -0.1],
        predicted_gain_mean=0.2,
        predicted_gain_std=0.05,
        risk_adjusted_score=0.15,
        rationale={},
        execution={},
    )
    assert result.delta_q == [0.1, -0.1]
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_slice_remix_types -v`

Expected: FAIL because `slice_remix` package does not exist.

**Step 3: Write minimal implementation**

Create dataclasses for:
- `RemixContext`
- `CandidateAction`
- `ResponseRow`
- `RecommendationResult`

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_slice_remix_types -v`

Expected: PASS

**Step 5: Commit**

```bash
git add slice_remix/__init__.py slice_remix/types.py tests/test_slice_remix_types.py
git commit -m "feat: add slice remix type definitions"
```

### Task 2: Load slice artifacts and estimate baseline mixtures

**Files:**
- Create: `slice_remix/baseline.py`
- Test: `tests/test_slice_remix_baseline.py`

**Step 1: Write the failing test**

```python
import numpy as np

from slice_remix.baseline import estimate_baseline_mixture


def test_estimate_baseline_mixture_averages_memberships():
    memberships = np.asarray(
        [
            [1.0, 0.0],
            [0.2, 0.8],
            [0.4, 0.6],
        ],
        dtype=np.float32,
    )
    mixture = estimate_baseline_mixture(memberships, sample_indices=[0, 2])
    assert np.allclose(mixture, [0.7, 0.3])
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_slice_remix_baseline -v`

Expected: FAIL because `estimate_baseline_mixture` does not exist.

**Step 3: Write minimal implementation**

Implement:
- membership subset selection
- mixture averaging
- validation that the result sums to 1 within tolerance

Also add a small helper that loads:
- `slice_result.npz`
- `slice_result_meta.json`

and extracts memberships / sample ids for downstream remix code.

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_slice_remix_baseline -v`

Expected: PASS

**Step 5: Commit**

```bash
git add slice_remix/baseline.py tests/test_slice_remix_baseline.py
git commit -m "feat: add baseline mixture estimation"
```

### Task 3: Implement sparse remix action generation

**Files:**
- Create: `slice_remix/actions.py`
- Test: `tests/test_slice_remix_actions.py`

**Step 1: Write the failing test**

```python
import numpy as np

from slice_remix.actions import generate_pairwise_candidates


def test_generate_pairwise_candidates_preserves_simplex():
    baseline = np.asarray([0.4, 0.35, 0.25], dtype=np.float32)
    candidates = generate_pairwise_candidates(
        baseline,
        amplitudes=[0.05],
        ordered_pairs=[(0, 1), (2, 1)],
    )
    assert len(candidates) == 2
    for candidate in candidates:
        assert np.isclose(candidate.target_mixture.sum(), 1.0)
        assert np.all(candidate.target_mixture >= 0.0)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_slice_remix_actions -v`

Expected: FAIL because candidate generation is missing.

**Step 3: Write minimal implementation**

Implement:
- pairwise transfer generation
- optional sparse general action generation stub
- action metadata:
  - donors
  - receivers
  - amplitude
  - support size
  - `delta_q`

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_slice_remix_actions -v`

Expected: PASS

**Step 5: Commit**

```bash
git add slice_remix/actions.py tests/test_slice_remix_actions.py
git commit -m "feat: add sparse remix candidate generation"
```

### Task 4: Compute feature-level slice portraits and portrait shifts

**Files:**
- Create: `slice_remix/portraits.py`
- Test: `tests/test_slice_remix_portraits.py`

**Step 1: Write the failing test**

```python
import numpy as np

from slice_remix.portraits import compute_slice_portraits, compute_portrait_shift


def test_compute_portrait_shift_from_slice_portraits():
    feature_groups = {
        "feature_a": np.asarray([[1.0], [3.0], [5.0]], dtype=np.float32),
    }
    memberships = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    portraits = compute_slice_portraits(feature_groups, memberships)
    baseline = np.asarray([0.5, 0.5], dtype=np.float32)
    target = np.asarray([0.7, 0.3], dtype=np.float32)
    shift = compute_portrait_shift(portraits, baseline, target)
    assert "feature_a" in shift
    assert shift["feature_a"].shape == (1,)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_slice_remix_portraits -v`

Expected: FAIL because portrait utilities are missing.

**Step 3: Write minimal implementation**

Implement:
- feature-group extraction from projected artifacts or report schema
- slice portrait computation
- baseline / target expected portrait computation
- `delta_phi` computation per feature group

Keep the representation grouped by feature name, not flattened too early.

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_slice_remix_portraits -v`

Expected: PASS

**Step 5: Commit**

```bash
git add slice_remix/portraits.py tests/test_slice_remix_portraits.py
git commit -m "feat: add feature portrait shift computation"
```

### Task 5: Implement sample-level execution policy

**Files:**
- Create: `slice_remix/policy.py`
- Test: `tests/test_slice_remix_policy.py`

**Step 1: Write the failing test**

```python
import numpy as np

from slice_remix.policy import compute_importance_weights


def test_importance_weights_increase_for_upweighted_slice():
    memberships = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    target = np.asarray([0.7, 0.3], dtype=np.float32)
    weights = compute_importance_weights(memberships, target)
    assert weights[0] > weights[1]
    assert np.isclose(weights.sum(), 1.0)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_slice_remix_policy -v`

Expected: FAIL because the execution policy is missing.

**Step 3: Write minimal implementation**

Implement the practical weighting approximation:
- compute pool mixture
- compute slice reweight multipliers
- compute sample scores
- normalize into probabilities

Also add a helper that summarizes expected per-slice training quotas under fixed budget.

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_slice_remix_policy -v`

Expected: PASS

**Step 5: Commit**

```bash
git add slice_remix/policy.py tests/test_slice_remix_policy.py
git commit -m "feat: add budgeted execution policy"
```

### Task 6: Define the response dataset schema

**Files:**
- Create: `slice_remix/dataset.py`
- Test: `tests/test_slice_remix_dataset.py`

**Step 1: Write the failing test**

```python
from slice_remix.dataset import build_response_row


def test_build_response_row_contains_core_fields():
    row = build_response_row(
        baseline_trial_id="trial_0",
        candidate_id="cand_1",
        baseline_mixture=[0.5, 0.5],
        target_mixture=[0.6, 0.4],
        delta_q=[0.1, -0.1],
        delta_phi={"feature_a": [0.2]},
        context={"budget": 1000},
        measured_gain=0.3,
    )
    assert row["baseline_trial_id"] == "trial_0"
    assert row["context"]["budget"] == 1000
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_slice_remix_dataset -v`

Expected: FAIL because dataset row creation is missing.

**Step 3: Write minimal implementation**

Implement:
- `build_response_row`
- JSON-serializable conversion for NumPy values
- optional helpers for reading/writing `.jsonl`

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_slice_remix_dataset -v`

Expected: PASS

**Step 5: Commit**

```bash
git add slice_remix/dataset.py tests/test_slice_remix_dataset.py
git commit -m "feat: add remix response dataset schema"
```

### Task 7: Implement the first surrogate baseline

**Files:**
- Create: `slice_remix/surrogate.py`
- Test: `tests/test_slice_remix_surrogate.py`

**Step 1: Write the failing test**

```python
from slice_remix.surrogate import LinearRemixSurrogate


def test_linear_surrogate_fits_and_predicts():
    model = LinearRemixSurrogate()
    rows = [
        {"delta_q": [0.1, -0.1], "delta_phi": {"feature_a": [0.2]}, "context": {"budget": 1000}, "measured_gain": 0.3},
        {"delta_q": [-0.1, 0.1], "delta_phi": {"feature_a": [-0.2]}, "context": {"budget": 1000}, "measured_gain": -0.2},
    ]
    model.fit(rows)
    pred = model.predict_mean(rows[:1])[0]
    assert isinstance(pred, float)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_slice_remix_surrogate -v`

Expected: FAIL because the surrogate is missing.

**Step 3: Write minimal implementation**

Implement:
- feature flattening for grouped portrait input
- a lightweight regularized linear fit using NumPy least squares / ridge-style solve
- `fit`
- `predict_mean`
- a placeholder `predict_std` that returns zeros in v1

Do not implement the quadratic term yet; get the linear baseline stable first.

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_slice_remix_surrogate -v`

Expected: PASS

**Step 5: Commit**

```bash
git add slice_remix/surrogate.py tests/test_slice_remix_surrogate.py
git commit -m "feat: add linear remix surrogate baseline"
```

### Task 8: Implement runtime recommendation ranking

**Files:**
- Create: `slice_remix/recommender.py`
- Test: `tests/test_slice_remix_recommender.py`

**Step 1: Write the failing test**

```python
import numpy as np

from slice_remix.recommender import rank_candidates


class StubSurrogate:
    def predict_mean(self, rows):
        return [row["measured_gain_hint"] for row in rows]

    def predict_std(self, rows):
        return [0.0 for _ in rows]


def test_rank_candidates_prefers_high_gain_low_complexity():
    candidates = [
        {"candidate_id": "a", "measured_gain_hint": 0.5, "l1_shift": 0.2, "support_size": 2},
        {"candidate_id": "b", "measured_gain_hint": 0.4, "l1_shift": 0.1, "support_size": 2},
    ]
    ranked = rank_candidates(candidates, StubSurrogate(), kappa=0.0, lambda_l1=0.0, lambda_support=0.0)
    assert ranked[0]["candidate_id"] == "a"
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_slice_remix_recommender -v`

Expected: FAIL because ranking is missing.

**Step 3: Write minimal implementation**

Implement:
- runtime score computation
- sorting
- structured recommendation object creation with:
  - baseline mixture
  - target mixture
  - delta_q
  - predicted gain
  - rationale summary
  - execution summary

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_slice_remix_recommender -v`

Expected: PASS

**Step 5: Commit**

```bash
git add slice_remix/recommender.py tests/test_slice_remix_recommender.py
git commit -m "feat: add runtime remix recommendation ranking"
```

### Task 9: Add a CLI to build response-dataset rows from slice artifacts

**Files:**
- Create: `run_remix_response_dataset.py`
- Test: `tests/test_remix_response_dataset_cli.py`

**Step 1: Write the failing test**

```python
from clip_dinoiser.run_remix_response_dataset import build_parser


def test_response_dataset_cli_parser_accepts_required_args():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--projected-dir", "/tmp/projected",
            "--cluster-dir", "/tmp/cluster",
            "--output-path", "/tmp/rows.jsonl",
            "--budget", "1000",
        ]
    )
    assert args.budget == 1000
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_remix_response_dataset_cli -v`

Expected: FAIL because the CLI does not exist.

**Step 3: Write minimal implementation**

Implement a CLI that:
- loads slice artifacts,
- samples baseline trials,
- generates candidate mixtures,
- computes `delta_q`, `delta_phi`, and execution summaries,
- writes JSONL rows without measured gains yet (label can be null in this stage).

This script is for experiment preparation, not model training.

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_remix_response_dataset_cli -v`

Expected: PASS

**Step 5: Commit**

```bash
git add run_remix_response_dataset.py tests/test_remix_response_dataset_cli.py
git commit -m "feat: add remix response dataset preparation cli"
```

### Task 10: Add a CLI to run runtime recommendation from fitted rows

**Files:**
- Create: `run_remix_recommendation.py`
- Test: `tests/test_remix_recommendation_cli.py`

**Step 1: Write the failing test**

```python
from clip_dinoiser.run_remix_recommendation import build_parser


def test_recommendation_cli_parser_accepts_required_args():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--projected-dir", "/tmp/projected",
            "--cluster-dir", "/tmp/cluster",
            "--response-dataset", "/tmp/rows.jsonl",
            "--baseline-seed", "0",
            "--budget", "1000",
        ]
    )
    assert args.baseline_seed == 0
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_remix_recommendation_cli -v`

Expected: FAIL because the CLI does not exist.

**Step 3: Write minimal implementation**

Implement a CLI that:
- loads response rows with measured gains,
- fits the surrogate,
- builds runtime candidates for a chosen baseline seed,
- ranks them,
- writes a recommendation JSON artifact.

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_remix_recommendation_cli -v`

Expected: PASS

**Step 5: Commit**

```bash
git add run_remix_recommendation.py tests/test_remix_recommendation_cli.py
git commit -m "feat: add remix recommendation cli"
```

### Task 11: Add an integration-style smoke test over synthetic artifacts

**Files:**
- Create: `tests/test_remix_pipeline_smoke.py`

**Step 1: Write the failing test**

Create a synthetic projected/cluster artifact fixture and verify:
- baseline estimation works,
- candidates are generated,
- portrait shifts are computed,
- response rows can be built,
- a surrogate can fit,
- a recommendation result is produced.

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_remix_pipeline_smoke -v`

Expected: FAIL because some pieces are still missing.

**Step 3: Write minimal implementation glue**

Patch any missing serialization / field naming inconsistencies surfaced by the smoke test.

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_remix_pipeline_smoke -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_remix_pipeline_smoke.py slice_remix run_remix_response_dataset.py run_remix_recommendation.py
git commit -m "test: add remix recommendation smoke coverage"
```

### Task 12: Run the focused test suite and document how to use the baseline

**Files:**
- Modify: `README.md`

**Step 1: Write a small README section**

Document:
- what `slice_remix` does,
- required inputs,
- how to prepare response rows,
- how to run the first recommendation baseline.

**Step 2: Run the focused suite**

Run:

```bash
python3 -m unittest \
  tests.test_slice_remix_types \
  tests.test_slice_remix_baseline \
  tests.test_slice_remix_actions \
  tests.test_slice_remix_portraits \
  tests.test_slice_remix_policy \
  tests.test_slice_remix_dataset \
  tests.test_slice_remix_surrogate \
  tests.test_slice_remix_recommender \
  tests.test_remix_response_dataset_cli \
  tests.test_remix_recommendation_cli \
  tests.test_remix_pipeline_smoke \
  -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document remix recommendation baseline workflow"
```
