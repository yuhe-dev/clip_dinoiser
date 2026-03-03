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


@torch.no_grad()
def extract(img_dir, out_dir="cache", n=200, seed=0, bs=64,
            model_name="ViT-B-32", pretrained="openai"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()

    paths = stable_sample_images(img_dir, n, seed)
    embs, kept = [], []

    for i in tqdm(range(0, len(paths), bs)):
        batch = paths[i:i+bs]
        imgs = [preprocess(Image.open(p).convert("RGB")) for p in batch]
        x = torch.stack(imgs).to(device)
        z = model.encode_image(x).float()
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-12)
        embs.append(z.cpu().numpy().astype(np.float32))
        kept.extend(batch)

    Z = np.concatenate(embs, axis=0)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "visual_emb_200.npy"), Z)
    with open(os.path.join(out_dir, "clip_paths.json"), "w") as f:
        json.dump(kept, f)

    print("saved", Z.shape)


if __name__ == "__main__":
    extract("./data/coco_stuff164k/images/train2017")