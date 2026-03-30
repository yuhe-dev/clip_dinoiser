from __future__ import annotations

import argparse
import faulthandler
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
    from slice_remix.class_coverage import load_class_presence_matrix, select_focus_class_spec
    from slice_remix.dataset import write_jsonl
    from slice_remix.portraits import build_feature_label_map, load_portrait_feature_groups
    from slice_remix.surrogate_features import build_surrogate_feature_payload
else:
    from .slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from .slice_discovery.projector import SliceFeatureProjector
    from .slice_remix.baseline import load_slice_artifacts
    from .slice_remix.class_coverage import load_class_presence_matrix, select_focus_class_spec
    from .slice_remix.dataset import write_jsonl
    from .slice_remix.portraits import build_feature_label_map, load_portrait_feature_groups
    from .slice_remix.surrogate_features import build_surrogate_feature_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare random-subset surrogate dataset rows from slice artifacts.")
    parser.add_argument("--projected-dir", required=True)
    parser.add_argument("--cluster-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--train-seeds", default="0:800")
    parser.add_argument("--val-seeds", default="800:900")
    parser.add_argument("--test-seeds", default="900:1000")
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--subset-manifest-dir")
    parser.add_argument("--pool-image-root")
    parser.add_argument("--portrait-source", choices=["auto", "projected", "semantic"], default="auto")
    parser.add_argument("--processed-data-root")
    parser.add_argument("--schema-path")
    parser.add_argument("--assembled-feature-dir")
    parser.add_argument("--annotation-root")
    parser.add_argument("--baseline-result-path")
    parser.add_argument("--full-result-path")
    parser.add_argument("--focus-class-gap-threshold", type=float, default=10.0)
    parser.add_argument("--focus-class-top-k", type=int, default=25)
    parser.add_argument("--include-hard-mixture", action="store_true")
    return parser


def _progress(message: str) -> None:
    print(f"[surrogate_random_dataset] {message}", file=sys.stderr, flush=True)


def parse_seed_spec(raw: str) -> list[int]:
    seeds: list[int] = []
    for token in str(raw).split(","):
        item = token.strip()
        if not item:
            continue
        if ":" in item:
            start_raw, end_raw = item.split(":", 1)
            start, end = int(start_raw), int(end_raw)
            if end < start:
                raise ValueError(f"invalid seed range '{item}'")
            seeds.extend(range(start, end))
        else:
            seeds.append(int(item))
    return seeds


def build_split_assignments(*, train_seeds: list[int], val_seeds: list[int], test_seeds: list[int]) -> list[tuple[int, str]]:
    split_map: dict[int, str] = {}
    ordered: list[tuple[int, str]] = []
    for split_name, seeds in [("train", train_seeds), ("val", val_seeds), ("test", test_seeds)]:
        for seed in seeds:
            if int(seed) in split_map:
                raise ValueError(f"seed {seed} assigned to multiple splits")
            split_map[int(seed)] = split_name
            ordered.append((int(seed), split_name))
    return sorted(ordered, key=lambda row: row[0])


def _load_optional_coverage_inputs(args: argparse.Namespace, sample_ids: list[str]) -> tuple[np.ndarray | None, dict[str, object] | None]:
    if not args.annotation_root:
        return None, None

    if not args.full_result_path:
        raise ValueError("--full-result-path is required when --annotation-root is provided")

    with open(args.full_result_path, "r", encoding="utf-8") as f:
        full_result = json.load(f)

    baseline_result = None
    if args.baseline_result_path:
        with open(args.baseline_result_path, "r", encoding="utf-8") as f:
            baseline_result = json.load(f)

    class_names = list(((full_result.get("coco_stuff") or {}).get("per_class") or {}).keys())
    if not class_names:
        raise ValueError("full result must contain coco_stuff.per_class entries")

    class_presence = load_class_presence_matrix(
        sample_ids=sample_ids,
        annotation_root=os.path.abspath(args.annotation_root),
        num_classes=len(class_names),
    )
    focus_spec = None
    if baseline_result is not None:
        focus_spec = select_focus_class_spec(
            baseline_result=baseline_result,
            full_result=full_result,
            min_iou_gap=float(args.focus_class_gap_threshold),
            top_k=int(args.focus_class_top_k),
        )
        if focus_spec is not None:
            focus_spec = dict(focus_spec)
            focus_spec["class_names"] = list(focus_spec.get("class_names", []))
    return class_presence, focus_spec


def _write_subset_manifest(
    manifest_dir: str,
    experiment_id: str,
    sample_ids: list[str],
    pool_image_root: str | None = None,
) -> str:
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, f"{experiment_id}.json")
    payload = {
        "candidate_id": experiment_id,
        "sample_ids": list(sample_ids),
    }
    if pool_image_root is not None:
        payload["sample_paths"] = [os.path.join(pool_image_root, sample_id) for sample_id in sample_ids]
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return manifest_path


def run(args: argparse.Namespace, log_fn=_progress) -> int:
    faulthandler.enable()
    log_fn("loading projected artifacts")
    projected = SliceFeatureProjector.load(os.path.abspath(args.projected_dir))
    log_fn("loading slice artifacts")
    artifacts = load_slice_artifacts(os.path.abspath(args.cluster_dir))
    if projected.sample_ids != artifacts.sample_ids:
        raise ValueError("projected sample ids must match cluster sample ids")

    log_fn(f"loading portrait feature groups source={args.portrait_source}")
    feature_groups, portrait_source = load_portrait_feature_groups(
        projected=projected,
        cluster_meta=artifacts.meta,
        portrait_source=args.portrait_source,
        processed_data_root=os.path.abspath(args.processed_data_root) if args.processed_data_root else None,
        schema_path=os.path.abspath(args.schema_path) if args.schema_path else None,
        assembled_feature_dir=os.path.abspath(args.assembled_feature_dir) if args.assembled_feature_dir else None,
        log_fn=log_fn,
    )
    feature_label_map = build_feature_label_map(
        feature_groups,
        schema_path=os.path.abspath(args.schema_path) if args.schema_path else None,
    )

    class_presence, focus_spec = _load_optional_coverage_inputs(args, artifacts.sample_ids)
    split_assignments = build_split_assignments(
        train_seeds=parse_seed_spec(args.train_seeds),
        val_seeds=parse_seed_spec(args.val_seeds),
        test_seeds=parse_seed_spec(args.test_seeds),
    )
    log_fn(f"preparing random subsets count={len(split_assignments)} budget={int(args.budget)} portrait_source={portrait_source}")

    rows: list[dict[str, object]] = []
    sample_count = len(artifacts.sample_ids)
    for subset_seed, split_name in split_assignments:
        rng = np.random.default_rng(int(subset_seed))
        sample_indices = rng.choice(sample_count, size=int(args.budget), replace=False)
        sample_indices_list = [int(index) for index in sample_indices.tolist()]
        sample_ids = [artifacts.sample_ids[index] for index in sample_indices_list]
        experiment_id = f"rand_subset_s{subset_seed:04d}_t{int(args.training_seed):02d}"

        payload = build_surrogate_feature_payload(
            feature_groups=feature_groups,
            sample_indices=sample_indices_list,
            memberships=artifacts.membership,
            hard_assignment=artifacts.hard_assignment,
            class_presence=class_presence,
            focus_class_indices=(focus_spec or {}).get("class_indices"),
            feature_label_map=feature_label_map,
            include_hard_mixture=bool(args.include_hard_mixture),
        )

        manifest_path = None
        if args.subset_manifest_dir:
            manifest_path = _write_subset_manifest(
                os.path.abspath(args.subset_manifest_dir),
                experiment_id,
                sample_ids,
                os.path.abspath(args.pool_image_root) if args.pool_image_root else None,
            )

        row = {
            "experiment_id": experiment_id,
            "source": "random_subset",
            "split": split_name,
            "subset_seed": int(subset_seed),
            "training_seed": int(args.training_seed),
            "budget": int(args.budget),
            "sample_ids": list(sample_ids),
            "sample_indices": sample_indices_list,
            "portrait_source": portrait_source,
            "manifest_path": manifest_path,
            "feature_payload": payload,
        }
        if focus_spec is not None:
            row["focus_class_names"] = list(focus_spec.get("class_names", []))
        rows.append(row)

    write_jsonl(os.path.abspath(args.output_path), rows)
    log_fn(f"wrote surrogate dataset rows={len(rows)} path={os.path.abspath(args.output_path)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
