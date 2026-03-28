from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

if __package__ in {None, ""}:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from slice_discovery.projector import SliceFeatureProjector
    from slice_remix.baseline import load_slice_artifacts
    from slice_remix.portraits import build_feature_label_map, load_portrait_feature_groups
    from slice_remix.prior_graph import PriorGraphHyperparams, build_prior_graph
else:
    from .slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from .slice_discovery.projector import SliceFeatureProjector
    from .slice_remix.baseline import load_slice_artifacts
    from .slice_remix.portraits import build_feature_label_map, load_portrait_feature_groups
    from .slice_remix.prior_graph import PriorGraphHyperparams, build_prior_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a prior graph artifact for the SliceTune workbench.")
    parser.add_argument("--projected-dir", required=True)
    parser.add_argument("--cluster-dir", required=True)
    parser.add_argument("--baseline-seed", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--portrait-source", choices=["auto", "projected", "semantic"], default="auto")
    parser.add_argument("--processed-data-root")
    parser.add_argument("--schema-path")
    parser.add_argument("--assembled-feature-dir")
    parser.add_argument("--top-k-render", type=int, default=12)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    return parser


def _progress(message: str) -> None:
    print(f"[prior_graph_export] {message}", file=sys.stderr, flush=True)


def run(args: argparse.Namespace, log_fn=_progress) -> int:
    projected = SliceFeatureProjector.load(os.path.abspath(args.projected_dir))
    artifacts = load_slice_artifacts(os.path.abspath(args.cluster_dir))
    if projected.sample_ids != artifacts.sample_ids:
        raise ValueError("projected sample ids must match cluster sample ids")

    feature_groups, portrait_source = load_portrait_feature_groups(
        projected=projected,
        cluster_meta=artifacts.meta,
        portrait_source=args.portrait_source,
        processed_data_root=os.path.abspath(args.processed_data_root) if args.processed_data_root else None,
        schema_path=os.path.abspath(args.schema_path) if args.schema_path else None,
        assembled_feature_dir=os.path.abspath(args.assembled_feature_dir) if args.assembled_feature_dir else None,
        log_fn=log_fn,
    )
    label_map = build_feature_label_map(
        feature_groups,
        schema_path=os.path.abspath(args.schema_path) if args.schema_path else None,
    )

    rng = np.random.default_rng(int(args.baseline_seed))
    sample_indices = rng.choice(len(artifacts.sample_ids), size=int(args.budget), replace=False)
    hyperparams = PriorGraphHyperparams(
        top_k_render=int(args.top_k_render),
        score_threshold=float(args.score_threshold),
    )
    payload = build_prior_graph(
        feature_groups=feature_groups,
        feature_label_map=label_map,
        memberships=artifacts.membership,
        baseline_sample_indices=sample_indices.tolist(),
        slice_ids=[f"slice_{index:02d}" for index in range(artifacts.membership.shape[1])],
        hyperparams=hyperparams,
        baseline_seed=int(args.baseline_seed),
        budget=int(args.budget),
    )

    result = payload.to_dict()
    result["graph_context"]["portrait_source"] = portrait_source

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log_fn(f"wrote prior graph to {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
