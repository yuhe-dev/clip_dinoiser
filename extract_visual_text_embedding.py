import os, json, random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import open_clip


def stable_sample_images(img_dir, n, seed=0):
    paths = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    rng = random.Random(seed)
    return rng.sample(paths, min(n, len(paths)))


def infer_mask_path(img_path: str, ann_dir: str, suffix: str = "_labelTrainIds.png"):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(ann_dir, stem + suffix)


def compute_area_weights(mask_np: np.ndarray, num_classes: int, ignore_index: int = 255,
                         class_ids=None):
    m = mask_np.astype(np.int32)
    valid = (m != ignore_index)
    if not valid.any():
        return np.zeros((num_classes,), np.float32), []

    vals = m[valid].reshape(-1)

    if class_ids is not None:
        counts = np.zeros((num_classes,), dtype=np.float32)
        for cid in class_ids:
            counts[int(cid)] = float((vals == int(cid)).sum())
    else:
        counts = np.bincount(vals, minlength=num_classes).astype(np.float32)

    total = float(counts.sum())
    if total <= 0:
        return np.zeros((num_classes,), np.float32), []

    w = counts / total
    present = np.where(counts > 0)[0].tolist()
    return w.astype(np.float32), present


def bbox_from_binary_mask(binm: np.ndarray):
    ys, xs = np.where(binm)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return x0, y0, x1, y1


def expand_bbox(bbox, W, H, expand=0.1):
    x0, y0, x1, y1 = bbox
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    ex = int(bw * expand)
    ey = int(bh * expand)
    x0 = max(0, x0 - ex)
    y0 = max(0, y0 - ey)
    x1 = min(W - 1, x1 + ex)
    y1 = min(H - 1, y1 + ey)
    return x0, y0, x1, y1


@torch.no_grad()
def extract_region_level_gap(
    img_dir,
    ann_dir,
    out_dir="cache_region_gap",
    n=20000,
    seed=0,
    bs=64,
    model_name="ViT-B-32",
    pretrained="openai",
    num_classes=171,
    ignore_index=255,
    use_things_only=True,
    thing_ids=range(0, 80),
    prompt_template="a photo of a {}",
    class_names=None,  # list[str] length=num_classes
    bbox_expand=0.1,
    min_region_pixels=128,
    max_regions_per_image=100,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    # ---- 1) class text embedding table T[c] ----
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    prompts = [prompt_template.format(name) for name in class_names]
    text_tokens = tokenizer(prompts).to(device)
    T = model.encode_text(text_tokens).float()             # [C,D]
    T = T / (T.norm(dim=-1, keepdim=True) + 1e-12)
    T_np = T.cpu().numpy().astype(np.float32)

    # ---- 2) sample images ----
    paths = stable_sample_images(img_dir, n, seed)
    class_ids = list(thing_ids) if use_things_only else None

    kept = []
    region_gap_scores = []     # one scalar per image
    debug_jsonl = []           # optional detailed info

    for i in tqdm(range(0, len(paths), bs), desc="Region-level gap"):
        batch = paths[i:i+bs]

        # collect valid samples
        samples = []
        for p in batch:
            mp = infer_mask_path(p, ann_dir, suffix="_labelTrainIds.png")
            if not os.path.exists(mp):
                continue
            try:
                img = Image.open(p).convert("RGB")
                mask_np = np.array(Image.open(mp))
            except Exception:
                continue
            samples.append((p, img, mask_np))

        if not samples:
            continue

        # ---- compute region-level gap per image (sequential per image; crop count varies) ----
        for p, img, mask_np in samples:
            m = mask_np.astype(np.int32)
            H, W = m.shape[:2]

            w, present = compute_area_weights(
                mask_np, num_classes=num_classes, ignore_index=ignore_index, class_ids=class_ids
            )
            if len(present) == 0:
                continue

            # build candidate regions: (cid, area, bbox)
            regions = []
            for cid in present:
                binm = (m == int(cid))
                area = int(binm.sum())
                if area < min_region_pixels:
                    continue
                bbox = bbox_from_binary_mask(binm)
                if bbox is None:
                    continue
                bbox = expand_bbox(bbox, W=W, H=H, expand=bbox_expand)
                regions.append((int(cid), area, bbox))

            if not regions:
                continue

            # keep largest regions for speed
            regions.sort(key=lambda x: x[1], reverse=True)
            regions = regions[:max_regions_per_image]

            # compute crop embeddings in a mini-batch (better GPU utilization)
            crops = []
            cids = []
            areas = []
            for cid, area, (x0, y0, x1, y1) in regions:
                crop = img.crop((x0, y0, x1 + 1, y1 + 1))  # PIL crop uses (left, upper, right, lower)
                crops.append(preprocess(crop))
                cids.append(cid)
                areas.append(area)

            x = torch.stack(crops).to(device)  # [R,3,h,w] (after preprocess becomes fixed size)
            v = model.encode_image(x).float()   # [R,D]
            v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
            v_np = v.cpu().numpy().astype(np.float32)

            # per-region gap with class text embedding
            gaps = []
            weights = []
            for k, cid in enumerate(cids):
                cos = float(np.dot(v_np[k], T_np[cid]))
                gap = 1.0 - cos
                gaps.append(gap)
                # use area fraction over selected regions or over all selected classes?
                # 推荐：用该类在整图里像素占比 w[cid]
                weights.append(float(w[cid]))

            weights = np.array(weights, dtype=np.float32)
            if weights.sum() <= 0:
                # fallback: normalize by region areas
                weights = np.array(areas, dtype=np.float32)
                weights = weights / (weights.sum() + 1e-12)
            else:
                weights = weights / (weights.sum() + 1e-12)

            gap_i = float(np.sum(weights * np.array(gaps, dtype=np.float32)))

            kept.append(p)
            region_gap_scores.append(gap_i)

            debug_jsonl.append({
                "path": p,
                "gap": gap_i,
                "n_regions": len(cids),
                "regions": [
                    {"cid": int(cids[k]), "area": int(areas[k]), "gap": float(gaps[k]), "w": float(weights[k])}
                    for k in range(len(cids))
                ]
            })

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "clip_paths.json"), "w") as f:
        json.dump(kept, f)
    np.save(os.path.join(out_dir, "region_gap.npy"), np.array(region_gap_scores, dtype=np.float32))
    np.save(os.path.join(out_dir, "clip_class_text_emb.npy"), T_np)

    # optional debug
    with open(os.path.join(out_dir, "region_gap_debug.jsonl"), "w") as f:
        for row in debug_jsonl:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({
            "model_name": model_name,
            "pretrained": pretrained,
            "prompt_template": prompt_template,
            "use_things_only": use_things_only,
            "num_classes": num_classes,
            "ignore_index": ignore_index,
            "bbox_expand": bbox_expand,
            "min_region_pixels": min_region_pixels,
            "max_regions_per_image": max_regions_per_image,
            "n": n,
            "seed": seed,
        }, f, ensure_ascii=False, indent=2)

    print("saved region_gap:", (len(region_gap_scores),))
    print("saved paths:", len(kept))
    print("saved class_text_emb:", T_np.shape)


if __name__ == "__main__":
    img_dir = "./data/coco_stuff164k/images/train2017"
    ann_dir = "./data/coco_stuff164k/annotations/train2017"

    from segmentation.datasets.coco_stuff import COCOStuffDataset
    class_names = list(COCOStuffDataset.CLASSES)

    extract_region_level_gap(
        img_dir, ann_dir,
        out_dir="cache_region_gap",
        n=200, bs=10, seed=0,
        class_names=class_names,
        use_things_only=True,
        min_region_pixels=128,
        bbox_expand=0.1,
        max_regions_per_image=100
    )