import argparse
from pathlib import Path
from typing import Dict, List, Sequence

from clip_dinoiser.feature_utils.data_feature.dataset_specs import get_dataset_feature_spec
from clip_dinoiser.tools.sample_coco_stuff_subset import (
    materialize_subset,
    sample_records,
    write_metadata_files,
)


def build_split_records(
    source_root: Path,
    split_path: Path,
    *,
    image_dir: str = "JPEGImages",
    annotation_dir: str = "SegmentationClass",
    image_suffix: str = ".jpg",
    annotation_suffix: str = ".png",
) -> List[Dict[str, str]]:
    source_root = Path(source_root)
    split_path = Path(split_path)
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    records: List[Dict[str, str]] = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        stem = line.strip()
        if not stem:
            continue
        image_path = source_root / image_dir / f"{stem}{image_suffix}"
        annotation_path = source_root / annotation_dir / f"{stem}{annotation_suffix}"
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found for split entry '{stem}': {image_path}")
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Annotation not found for split entry '{stem}': {annotation_path}")
        records.append(
            {
                "stem": stem,
                "image_rel": image_path.relative_to(source_root).as_posix(),
                "annotation_rel": annotation_path.relative_to(source_root).as_posix(),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a deterministic VOC20 segmentation subset and materialize it under data/."
    )
    parser.add_argument(
        "--source-root",
        default="data/VOCdevkit/VOC2012",
        help="Path to the source VOC2012 root.",
    )
    parser.add_argument(
        "--subset-root",
        default="data/voc20_subset_seed0",
        help="Path to the output subset root.",
    )
    parser.add_argument(
        "--split-path",
        default="ImageSets/Segmentation/train.txt",
        help="Split file relative to source root.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of image/annotation pairs to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to materialize the sampled files inside the subset root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    subset_root = Path(args.subset_root).resolve()
    split_path = (source_root / args.split_path).resolve()
    spec = get_dataset_feature_spec("voc20")

    print(f"[subset] Source root: {source_root}")
    print(f"[subset] Split path: {split_path}")
    print(f"[subset] Subset root: {subset_root}")
    print(f"[subset] Sampling {args.sample_size} pairs with seed={args.seed}")

    records = build_split_records(
        source_root,
        split_path,
        image_suffix=str(spec.image_suffixes[0]),
        annotation_suffix=str(spec.annotation_suffix),
    )
    print(f"[subset] Matched {len(records)} split records")

    sampled_records = sample_records(records, sample_size=args.sample_size, seed=args.seed)
    print(f"[subset] Sampled {len(sampled_records)} deterministic pairs")

    materialize_subset(
        source_root=source_root,
        subset_root=subset_root,
        sampled_records=sampled_records,
        link_mode=args.link_mode,
        verbose=True,
    )
    write_metadata_files(
        subset_root=subset_root,
        source_root=source_root,
        sampled_records=sampled_records,
        seed=args.seed,
        sample_size=args.sample_size,
        split=str(args.split_path),
        link_mode=args.link_mode,
        dataset_name="voc20_subset",
        verbose=True,
    )
    print("[subset] Done.")


if __name__ == "__main__":
    main()
