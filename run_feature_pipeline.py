import argparse
import os

import numpy as np

from feature_utils.data_feature.pipeline import DataFeaturePipelineRunner


def build_argparser():
    parser = argparse.ArgumentParser(description="Run raw extraction, postprocess, or the full feature pipeline.")
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=["quality", "difficulty", "coverage"],
        choices=["quality", "difficulty", "coverage"],
    )
    parser.add_argument("--stage", choices=["raw", "postprocess", "full"], default="full")
    parser.add_argument("--subset-root", default="./data/coco_stuff50k")
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--data-root", default="./data/data_feature")
    parser.add_argument("--schema-path", default="./docs/feature_schema/unified_processed_feature_schema.json")
    parser.add_argument("--embedding-root", default="./data/data_feature/coverage/visual_embedding")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--knn-k", type=int, default=50)
    parser.add_argument("--prototype-top-m", type=int, default=50)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def load_subset_records(index_path):
    records = np.load(index_path, allow_pickle=True)
    return [dict(item) for item in records.tolist()]


def build_feature_meta(args, dimension_name):
    if dimension_name == "quality":
        return {"patch_size": args.patch_size, "stride": args.stride}
    if dimension_name == "coverage":
        return {
            "embedding_root": args.embedding_root,
            "knn_k": int(args.knn_k),
            "prototype_top_m": int(args.prototype_top_m),
            "embeddings_file": "visual_emb.npy",
            "paths_file": "clip_paths_abs.json",
            "centroid_file": "prototypes_k200.npy",
            "knn_metric": "cosine",
            "normalize_for_cosine": True,
        }
    return {}


def main():
    args = build_argparser().parse_args()
    runner = DataFeaturePipelineRunner()
    subset_root = os.path.abspath(args.subset_root)
    index_path = os.path.abspath(args.index_path or os.path.join(subset_root, "sample_index.npy"))
    subset_records = None
    if args.stage in {"raw", "full"}:
        subset_records = load_subset_records(index_path)

    for dimension_name in args.dimensions:
        feature_meta = build_feature_meta(args, dimension_name)
        if args.stage == "raw":
            runner.run_raw(
                dimension_name=dimension_name,
                subset_root=subset_root,
                subset_records=subset_records,
                data_root=os.path.abspath(args.data_root),
                index_path=index_path,
                feature_meta=feature_meta,
                progress_interval=int(args.progress_interval),
                show_progress=not args.no_progress,
            )
        elif args.stage == "postprocess":
            runner.run_postprocess(
                dimension_name=dimension_name,
                data_root=os.path.abspath(args.data_root),
                schema_path=os.path.abspath(args.schema_path),
                progress_interval=int(args.progress_interval),
            )
        else:
            runner.run_full(
                dimension_name=dimension_name,
                subset_root=subset_root,
                subset_records=subset_records,
                data_root=os.path.abspath(args.data_root),
                index_path=index_path,
                feature_meta=feature_meta,
                schema_path=os.path.abspath(args.schema_path),
                progress_interval=int(args.progress_interval),
                show_progress=not args.no_progress,
            )


if __name__ == "__main__":
    main()
