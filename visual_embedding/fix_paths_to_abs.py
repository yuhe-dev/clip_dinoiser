# fix_paths_to_abs.py
import json
from pathlib import Path

root = Path("..").resolve()  # visual_embedding 的上一级就是项目根
paths = json.load(open("clip_paths.json","r"))

abs_paths = []
for p in paths:
    pth = Path(p)
    abs_paths.append(str((root / pth).resolve()) if not pth.is_absolute() else str(pth))

json.dump(abs_paths, open("clip_paths_abs.json","w"))
print("saved", len(abs_paths))