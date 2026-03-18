python -m torch.distributed.run --nproc_per_node=1 \
  run_remix_training_experiment.py \
  --config feature_experiment_fast \
  --subset-manifest ./artifacts/remix_seed3_eval/manifests/baseline_3.json \
  --output-dir ./artifacts/remix_seed3_eval/runs/baseline_3 \
  --result-name result.json
python -m torch.distributed.run --nproc_per_node=1 \
  run_remix_training_experiment.py \
  --config feature_experiment_fast \
  --subset-manifest ./artifacts/remix_seed3_eval/manifests/recommended_seed3.json \
  --output-dir ./artifacts/remix_seed3_eval/runs/recommended_seed3 \
  --result-name result.json
python run_remix_analysis_report.py \
  --response-dataset ./artifacts/remix_mvp_gmm_k8_b1000/rows_labeled.jsonl \
  --recommendation-path ./artifacts/remix_seed3_eval/recommendation_seed3.json \
  --baseline-result-path ./artifacts/remix_seed3_eval/runs/baseline_3/result.json \
  --recommended-result-path ./artifacts/remix_seed3_eval/runs/recommended_seed3/result.json \
  --metric-path coco_stuff.summary.mIoU \
  --output-path ./artifacts/remix_seed3_eval/analysis_seed3.json