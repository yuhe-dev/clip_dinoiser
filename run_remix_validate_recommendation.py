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
    from slice_remix.baseline import load_slice_artifacts
    from slice_remix.class_coverage import load_class_presence_matrix, select_focus_class_spec
    from slice_remix.manifests import load_subset_manifest
    from slice_remix.policy import compute_importance_weights, materialize_budgeted_subset
else:
    from .slice_remix.baseline import load_slice_artifacts
    from .slice_remix.class_coverage import load_class_presence_matrix, select_focus_class_spec
    from .slice_remix.manifests import load_subset_manifest
    from .slice_remix.policy import compute_importance_weights, materialize_budgeted_subset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize a recommendation into a manifest for validation experiments.")
    parser.add_argument("--cluster-dir", required=True)
    parser.add_argument("--recommendation-path", required=True)
    parser.add_argument("--pool-image-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--annotation-root")
    parser.add_argument("--baseline-result-path")
    parser.add_argument("--full-result-path")
    parser.add_argument("--focus-class-gap-threshold", type=float, default=10.0)
    parser.add_argument("--focus-class-top-k", type=int, default=25)
    parser.add_argument("--coverage-alpha", type=float, default=0.25)
    parser.add_argument("--coverage-repair-budget", type=int, default=64)
    return parser


def _progress(message: str) -> None:
    print(f"[remix_validate_recommendation] {message}", file=sys.stderr, flush=True)


def _load_optional_coverage_inputs(args: argparse.Namespace, sample_ids: list[str]) -> tuple[np.ndarray | None, dict[str, object] | None]:
    if not (args.annotation_root and args.baseline_result_path and args.full_result_path):
        return None, None
    with open(os.path.abspath(args.baseline_result_path), "r", encoding="utf-8") as f:
        baseline_result = json.load(f)
    with open(os.path.abspath(args.full_result_path), "r", encoding="utf-8") as f:
        full_result = json.load(f)
    focus_spec = select_focus_class_spec(
        baseline_result=baseline_result,
        full_result=full_result,
        min_iou_gap=float(args.focus_class_gap_threshold),
        top_k=int(args.focus_class_top_k),
    )
    if not focus_spec["class_indices"]:
        return None, focus_spec
    class_names = list(((full_result.get("coco_stuff") or {}).get("per_class") or {}).keys())
    class_presence = load_class_presence_matrix(
        sample_ids=sample_ids,
        annotation_root=os.path.abspath(args.annotation_root),
        num_classes=len(class_names),
    )
    return class_presence, focus_spec


def run(args: argparse.Namespace) -> int:
    cluster_dir = os.path.abspath(args.cluster_dir)
    recommendation_path = os.path.abspath(args.recommendation_path)
    pool_image_root = os.path.abspath(args.pool_image_root)
    output_manifest = os.path.abspath(args.output_manifest)

    _progress("loading recommendation")
    with open(recommendation_path, "r", encoding="utf-8") as f:
        recommendation = json.load(f)

    _progress("loading slice artifacts")
    artifacts = load_slice_artifacts(cluster_dir)
    target_mixture = np.asarray(recommendation["target_mixture"], dtype=np.float32)
    context = recommendation.get("context", {})
    execution = recommendation.get("execution", {})
    budget = int(context.get("budget", len(artifacts.sample_ids)))
    selection_seed = int(args.seed) if args.seed is not None else int(execution.get("selection_seed", 0))

    class_presence, focus_spec = _load_optional_coverage_inputs(args, artifacts.sample_ids)

    _progress(f"computing recommendation subset budget={budget} selection_seed={selection_seed}")
    weights = compute_importance_weights(artifacts.membership, target_mixture)
    materialized = materialize_budgeted_subset(
        artifacts.sample_ids,
        weights,
        budget=budget,
        seed=selection_seed,
        memberships=artifacts.membership,
        target_mixture=target_mixture,
        class_presence=class_presence,
        focus_class_indices=(focus_spec or {}).get("class_indices"),
        focus_class_weights=np.asarray((focus_spec or {}).get("class_weights", []), dtype=np.float32) if focus_spec else None,
        coverage_alpha=float(args.coverage_alpha),
        coverage_repair_budget=int(args.coverage_repair_budget),
    )
    selected_ids = materialized.selected_ids

    payload = {
        "candidate_id": recommendation.get("candidate_id"),
        "sample_ids": selected_ids,
        "sample_paths": [os.path.join(pool_image_root, sample_id) for sample_id in selected_ids],
        "source_recommendation": recommendation_path,
        "materialization": {
            "policy": "quota_mixture_coverage_v1" if class_presence is not None else "quota_mixture_v2",
            "focus_class_indices": list((focus_spec or {}).get("class_indices", [])),
            "focus_class_names": list((focus_spec or {}).get("class_names", [])),
            "mixture_l1_before_coverage_repair": materialized.mixture_l1_before_coverage_repair,
            "mixture_l1_after_coverage_repair": materialized.mixture_l1_after_coverage_repair,
            "focus_coverage_before": materialized.focus_coverage_before,
            "focus_coverage_after": materialized.focus_coverage_after,
            "accepted_coverage_swaps": materialized.accepted_coverage_swaps,
        },
    }
    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    _progress(f"wrote manifest path={output_manifest} sample_count={len(selected_ids)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
