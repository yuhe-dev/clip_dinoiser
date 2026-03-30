#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/yuhe/clip_dinoiser}
PYTHON=${PYTHON:-python}
MANIFEST_DIR=${MANIFEST_DIR:-$ROOT/artifacts/manual_target_review_k24_seed0/manifests_top10_policy_v3}
RUNS_DIR=${RUNS_DIR:-$ROOT/artifacts/manual_target_review_k24_seed0/runs_policy_v3}
CONFIG=${CONFIG:-feature_experiment_fast}

cd "$ROOT"
mkdir -p "$RUNS_DIR"

jobs=(
  "0 29501 baseline_seed0_budget1000"
  "1 29502 cand_progress_top05"
  "2 29503 cand_progress_top06"
  "3 29504 cand_progress_top01"
)

for job in "${jobs[@]}"; do
  read -r gpu port run_id <<< "$job"
  out_dir="$RUNS_DIR/$run_id"
  manifest="$MANIFEST_DIR/$run_id.json"

  if [[ -f "$out_dir/result.json" ]]; then
    echo "skip $run_id"
    continue
  fi

  mkdir -p "$out_dir"
  echo "[LAUNCH] gpu=$gpu port=$port run_id=$run_id"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4

    "$PYTHON" -m torch.distributed.run --nproc_per_node=1 --master_port "$port" \
      run_remix_training_experiment.py \
      --config "$CONFIG" \
      --subset-manifest "$manifest" \
      --output-dir "$out_dir" \
      --result-name result.json \
      > "$out_dir/stdout.log" 2>&1
  ) &
done

wait
echo "all jobs finished"
