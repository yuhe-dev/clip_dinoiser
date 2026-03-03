python -m torch.distributed.run --nproc_per_node=1 feature_experiment_pipeline.py knn_local_density_low.yaml
python -m torch.distributed.run --nproc_per_node=1 feature_experiment_pipeline.py knn_local_density_high.yaml
python -m torch.distributed.run --nproc_per_node=1 feature_experiment_pipeline.py knn_local_density_mixed.yaml
python -m torch.distributed.run --nproc_per_node=1 feature_experiment_pipeline.py knn_local_density_random.yaml