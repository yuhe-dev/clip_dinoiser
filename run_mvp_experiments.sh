for manifest in ./artifacts/remix_mvp_gmm_k8_b1000/manifests/*.json; do
  run_id=$(basename "${manifest%.json}")
  out_dir="./artifacts/remix_mvp_gmm_k8_b1000/runs/$run_id"

  if [ -f "$out_dir/result.json" ]; then
    echo "skip $run_id"
    continue
  fi

  python -m torch.distributed.run --nproc_per_node=1 \
    run_remix_training_experiment.py \
    --config feature_experiment_fast \
    --subset-manifest "$manifest" \
    --output-dir "$out_dir" \
    --result-name result.json || break
done