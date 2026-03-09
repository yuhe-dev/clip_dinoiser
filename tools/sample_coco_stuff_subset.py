import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


IMAGE_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
ANNOTATION_SUFFIX = "_labelTrainIds.png"


def build_pair_records(
    images_dir: Path,
    annotations_dir: Path,
    source_root: Path,
    image_suffixes: Sequence[str] = IMAGE_SUFFIXES,
    annotation_suffix: str = ANNOTATION_SUFFIX,
) -> List[Dict[str, str]]:
    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)
    source_root = Path(source_root)

    records: List[Dict[str, str]] = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in image_suffixes:
            continue

        stem = image_path.stem
        ann_path = annotations_dir / f"{stem}{annotation_suffix}"
        if not ann_path.is_file():
            continue

        records.append(
            {
                "stem": stem,
                "image_rel": image_path.relative_to(source_root).as_posix(),
                "annotation_rel": ann_path.relative_to(source_root).as_posix(),
            }
        )
    return records


def sample_records(
    records: Sequence[Dict[str, str]],
    sample_size: int,
    seed: int,
) -> List[Dict[str, str]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(records):
        raise ValueError(
            f"sample_size={sample_size} exceeds available records={len(records)}"
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(records), size=sample_size, replace=False)
    sampled = [records[int(index)] for index in sorted(indices.tolist())]
    return sampled


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def link_or_copy_file(src: Path, dst: Path, link_mode: str) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if link_mode == "copy":
        shutil.copy2(src, dst)
        return
    if link_mode == "hardlink":
        os.link(src, dst)
        return
    if link_mode == "symlink":
        rel_src = os.path.relpath(src, start=dst.parent)
        os.symlink(rel_src, dst)
        return
    raise ValueError(f"Unsupported link_mode='{link_mode}'")


def materialize_subset(
    source_root: Path,
    subset_root: Path,
    sampled_records: Sequence[Dict[str, str]],
    link_mode: str = "symlink",
    verbose: bool = True,
) -> None:
    source_root = Path(source_root)
    subset_root = Path(subset_root)
    subset_root.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[subset] Materializing {len(sampled_records)} samples into {subset_root}")

    for index, record in enumerate(sampled_records, start=1):
        image_src = source_root / record["image_rel"]
        ann_src = source_root / record["annotation_rel"]
        image_dst = subset_root / record["image_rel"]
        ann_dst = subset_root / record["annotation_rel"]

        link_or_copy_file(image_src, image_dst, link_mode=link_mode)
        link_or_copy_file(ann_src, ann_dst, link_mode=link_mode)

        if verbose and (index == 1 or index % 5000 == 0 or index == len(sampled_records)):
            print(f"[subset] Wrote {index}/{len(sampled_records)} sample pairs")


def write_metadata_files(
    subset_root: Path,
    source_root: Path,
    sampled_records: Sequence[Dict[str, str]],
    seed: int,
    sample_size: int,
    split: str,
    link_mode: str,
    verbose: bool = True,
) -> Tuple[Path, Path]:
    subset_root = Path(subset_root)
    subset_root.mkdir(parents=True, exist_ok=True)

    index_path = subset_root / "sample_index.npy"
    config_path = subset_root / "subset_config.json"

    records_array = np.asarray(list(sampled_records), dtype=object)
    np.save(index_path, records_array, allow_pickle=True)

    config = {
        "dataset_name": "coco_stuff164k_subset",
        "source_root": str(Path(source_root)),
        "subset_root": str(subset_root),
        "split": split,
        "sample_size": int(sample_size),
        "seed": int(seed),
        "link_mode": link_mode,
        "num_records": int(len(sampled_records)),
        "index_file": index_path.name,
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")

    if verbose:
        print(f"[subset] Saved metadata: {index_path}")
        print(f"[subset] Saved config: {config_path}")
        print("[subset] Example records:")
        for record in list(sampled_records)[:3]:
            print(f"  - {record}")

    return index_path, config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a deterministic COCO-Stuff train subset and materialize it under data/."
    )
    parser.add_argument(
        "--source-root",
        default="data/coco_stuff164k",
        help="Path to the source COCO-Stuff dataset root.",
    )
    parser.add_argument(
        "--subset-root",
        default="data/coco_stuff164k_subset_50k_seed0",
        help="Path to the output subset root.",
    )
    parser.add_argument(
        "--split",
        default="train2017",
        help="Dataset split to sample from.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
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
    images_dir = source_root / "images" / args.split
    annotations_dir = source_root / "annotations" / args.split

    print(f"[subset] Source root: {source_root}")
    print(f"[subset] Images dir: {images_dir}")
    print(f"[subset] Annotations dir: {annotations_dir}")
    print(f"[subset] Subset root: {subset_root}")
    print(f"[subset] Sampling {args.sample_size} pairs with seed={args.seed}")

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    records = build_pair_records(images_dir, annotations_dir, source_root)
    print(f"[subset] Matched {len(records)} image/annotation pairs")

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
        split=args.split,
        link_mode=args.link_mode,
        verbose=True,
    )
    print("[subset] Done.")


if __name__ == "__main__":
    main()
