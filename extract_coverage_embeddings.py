import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np


def load_subset_records(index_path: str) -> List[Dict[str, object]]:
    records = np.load(index_path, allow_pickle=True)
    return [dict(item) for item in records.tolist()]


def limit_subset_records(records: Sequence[Dict[str, object]], limit: int = None) -> List[Dict[str, object]]:
    if limit is None or int(limit) <= 0:
        return list(records)
    return list(records[: int(limit)])


def build_embedding_meta(
    clip_model: str,
    clip_pretrained: str,
    device: str,
    batch_size: int,
) -> Dict[str, object]:
    return {
        "clip_model": clip_model,
        "clip_pretrained": clip_pretrained,
        "device": device,
        "batch_size": int(batch_size),
    }


def resolve_subset_image_paths(
    subset_root: str,
    subset_records: Sequence[Dict[str, object]],
) -> Tuple[List[str], List[str]]:
    image_paths: List[str] = []
    image_rels: List[str] = []
    for record in subset_records:
        image_rel = str(record["image_rel"])
        image_paths.append(os.path.join(subset_root, image_rel))
        image_rels.append(image_rel)
    return image_paths, image_rels


def extract_subset_embeddings(
    subset_root: str,
    subset_records: Sequence[Dict[str, object]],
    embedding_meta: Dict[str, object],
    show_progress: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    from PIL import Image
    import torch
    import open_clip
    from tqdm import tqdm

    image_paths, _ = resolve_subset_image_paths(subset_root, subset_records)
    if not image_paths:
        return np.zeros((0, 0), dtype=np.float32), []

    clip_model = str(embedding_meta["clip_model"])
    clip_pretrained = str(embedding_meta["clip_pretrained"])
    device = str(embedding_meta["device"])
    batch_size = int(embedding_meta["batch_size"])

    model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=clip_pretrained)
    model = model.to(device).eval()

    embeddings: List[np.ndarray] = []
    kept_paths: List[str] = []
    iterator = tqdm(
        range(0, len(image_paths), batch_size),
        desc="Extracting coverage embeddings",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    with torch.no_grad():
        for start in iterator:
            batch_paths = image_paths[start : start + batch_size]
            batch_tensors = []
            valid_paths = []
            for image_path in batch_paths:
                if not os.path.exists(image_path):
                    continue
                image = Image.open(image_path).convert("RGB")
                batch_tensors.append(preprocess(image))
                valid_paths.append(os.path.abspath(image_path))
            if not batch_tensors:
                continue

            batch = torch.stack(batch_tensors).to(device)
            features = model.encode_image(batch).float()
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-12)
            embeddings.append(features.cpu().numpy().astype(np.float32))
            kept_paths.extend(valid_paths)

    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.concatenate(embeddings, axis=0), kept_paths


def save_embedding_assets(
    output_root: str,
    embeddings: np.ndarray,
    clip_paths: Sequence[str],
    subset_root: str,
    index_path: str,
    embedding_meta: Dict[str, object],
) -> Tuple[str, str, str]:
    os.makedirs(output_root, exist_ok=True)

    emb_path = os.path.join(output_root, "visual_emb.npy")
    paths_path = os.path.join(output_root, "clip_paths_abs.json")
    config_path = os.path.join(output_root, "coverage_embedding_config.json")

    np.save(emb_path, np.asarray(embeddings, dtype=np.float32))
    with open(paths_path, "w", encoding="utf-8") as f:
        json.dump(list(clip_paths), f, indent=2, ensure_ascii=False)

    config = {
        "subset_root": subset_root,
        "index_path": index_path,
        "embedding_meta": embedding_meta,
        "embedding_file": os.path.basename(emb_path),
        "paths_file": os.path.basename(paths_path),
        "num_samples": int(len(clip_paths)),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return emb_path, paths_path, config_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract CLIP embeddings for the COCO-Stuff 50k subset.")
    parser.add_argument("--subset-root", default="./data/coco_stuff50k")
    parser.add_argument("--index-path", default=None, help="Defaults to <subset-root>/sample_index.npy")
    parser.add_argument("--output-root", default="./data/data_feature/coverage/visual_embedding")
    parser.add_argument("--clip-model", default="ViT-B-16")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    subset_root = os.path.abspath(args.subset_root)
    index_path = os.path.abspath(args.index_path or os.path.join(subset_root, "sample_index.npy"))
    output_root = os.path.abspath(args.output_root)
    embedding_meta = build_embedding_meta(
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        device=args.device,
        batch_size=args.batch_size,
    )

    print(f"[coverage-embedding] subset_root={subset_root}")
    print(f"[coverage-embedding] index_path={index_path}")
    print(f"[coverage-embedding] output_root={output_root}")
    print(f"[coverage-embedding] embedding_meta={embedding_meta}")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Subset index not found: {index_path}")

    subset_records = load_subset_records(index_path)
    total_records = len(subset_records)
    subset_records = limit_subset_records(subset_records, limit=args.limit)
    print(f"[coverage-embedding] loaded {total_records} subset records")
    print(f"[coverage-embedding] processing {len(subset_records)} subset records (limit={args.limit})")

    embeddings, clip_paths = extract_subset_embeddings(
        subset_root=subset_root,
        subset_records=subset_records,
        embedding_meta=embedding_meta,
        show_progress=not args.no_progress,
    )
    emb_path, paths_path, config_path = save_embedding_assets(
        output_root=output_root,
        embeddings=embeddings,
        clip_paths=clip_paths,
        subset_root=subset_root,
        index_path=index_path,
        embedding_meta=embedding_meta,
    )

    print(f"[coverage-embedding] saved embeddings: {emb_path}")
    print(f"[coverage-embedding] saved paths: {paths_path}")
    print(f"[coverage-embedding] saved config: {config_path}")
    print(f"[coverage-embedding] embedding_shape={embeddings.shape}")


if __name__ == "__main__":
    main()
