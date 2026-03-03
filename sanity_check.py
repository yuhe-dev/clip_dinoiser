import json, numpy as np
T = np.load("cache_region_gap/clip_class_text_emb.npy")
gaps = np.load("cache_region_gap/region_gap.npy")
rows = [json.loads(l) for l in open("cache_region_gap/region_gap_debug.jsonl")]

r = rows[1]
print("saved gap", r["gap"], "array gap", gaps[1])
# 用 regions 里的 gap,w 重算
recon = sum(rr["gap"]*rr["w"] for rr in r["regions"])
print("recon gap", recon)

from segmentation.datasets.coco_stuff import COCOStuffDataset
names = list(COCOStuffDataset.CLASSES)

r = rows[3]
print(r["path"])
for rr in r["regions"]:
    print(rr["cid"], names[rr["cid"]], rr["area"], rr["gap"], rr["w"])