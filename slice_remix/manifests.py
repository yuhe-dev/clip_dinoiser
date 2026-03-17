from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class SubsetManifest:
    candidate_id: str | None
    sample_ids: list[str]
    sample_paths: list[str]


def load_subset_manifest(path: str, pool_image_root: str | None = None) -> SubsetManifest:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    sample_ids = [str(sample_id) for sample_id in payload.get("sample_ids", [])]
    sample_paths = payload.get("sample_paths")
    if sample_paths is None:
        if pool_image_root is None:
            raise ValueError("subset manifest requires sample_paths or pool_image_root")
        sample_paths = [os.path.join(pool_image_root, sample_id) for sample_id in sample_ids]
    return SubsetManifest(
        candidate_id=payload.get("candidate_id"),
        sample_ids=sample_ids,
        sample_paths=[str(sample_path) for sample_path in sample_paths],
    )
