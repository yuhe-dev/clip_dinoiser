#!/usr/bin/env bash
set -euo pipefail

MANIFEST_GLOB="./artifacts/remix_seed3_eval/manifests/cand_3_*.json"
RUNS_DIR="./artifacts/remix_seed3_eval/runs"
ROWS_PATH="./artifacts/remix_seed3_eval/rows_seed3_probe.jsonl"
RESULT_MANIFEST="./artifacts/remix_seed3_eval/result_manifest_seed3.jsonl"
ROWS_LABELED="./artifacts/remix_seed3_eval/rows_seed3_labeled.jsonl"
ROWS_COMBINED="./artifacts/remix_seed3_eval/rows_combined_with_seed3.jsonl"
RECOMMENDATION_PATH="./artifacts/remix_seed3_eval/recommendation_seed3.json"
BASELINE_RESULT="./artifacts/remix_seed3_eval/runs/baseline_3/result.json"
RECOMMENDED_RESULT="./artifacts/remix_seed3_eval/runs/recommended_seed3/result.json"
ANALYSIS_OUTPUT="./artifacts/remix_seed3_eval/analysis_seed3_full.json"
SEED012_ROWS="./artifacts/remix_mvp_gmm_k8_b1000/rows_labeled.jsonl"

mkdir -p "$RUNS_DIR"

echo "== Step 1: run seed=3 candidate training experiments =="
shopt -s nullglob
for manifest in $MANIFEST_GLOB; do
  run_id=$(basename "${manifest%.json}")
  out_dir="$RUNS_DIR/$run_id"

  if [ "$run_id" = "cand_3_7" ]; then
    echo "skip $run_id (already validated as recommended_seed3)"
    continue
  fi

  if [ -f "$out_dir/result.json" ]; then
    echo "skip $run_id"
    continue
  fi

  mkdir -p "$out_dir"
  echo "[RUN ] $run_id"

  python -m torch.distributed.run --nproc_per_node=1 \
    run_remix_training_experiment.py \
    --config feature_experiment_fast \
    --subset-manifest "$manifest" \
    --output-dir "$out_dir" \
    --result-name result.json
done
shopt -u nullglob

echo "== Step 2: collect result manifest =="
python run_remix_collect_results.py \
  --rows-path "$ROWS_PATH" \
  --results-dir "$RUNS_DIR" \
  --output-path "$RESULT_MANIFEST"

echo "== Step 3: attach results to rows =="
python run_remix_attach_results.py \
  --rows-path "$ROWS_PATH" \
  --result-manifest "$RESULT_MANIFEST" \
  --metric-path coco_stuff.summary.mIoU \
  --output-path "$ROWS_LABELED"

echo "== Step 4: combine seed 0/1/2 rows with seed 3 labeled rows =="
cat "$SEED012_ROWS" "$ROWS_LABELED" > "$ROWS_COMBINED"

echo "== Step 5: rerun full analysis report =="
python run_remix_analysis_report.py \
  --response-dataset "$ROWS_COMBINED" \
  --recommendation-path "$RECOMMENDATION_PATH" \
  --baseline-result-path "$BASELINE_RESULT" \
  --recommended-result-path "$RECOMMENDED_RESULT" \
  --metric-path coco_stuff.summary.mIoU \
  --output-path "$ANALYSIS_OUTPUT"

echo "== Done =="
echo "Full analysis written to: $ANALYSIS_OUTPUT"