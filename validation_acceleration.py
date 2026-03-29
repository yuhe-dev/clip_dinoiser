from __future__ import annotations

import os
import random
from typing import Any, Iterable
import re


class AttrDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _dataset_infos(dataset: object) -> tuple[list[dict[str, Any]], str]:
    if hasattr(dataset, "img_infos"):
        return list(getattr(dataset, "img_infos")), "img_infos"
    if hasattr(dataset, "data_infos"):
        return list(getattr(dataset, "data_infos")), "data_infos"
    raise RuntimeError("Dataset has no img_infos/data_infos.")


def _info_filename(info: dict[str, Any]) -> str | None:
    if "filename" in info:
        return str(info["filename"])
    if "img_path" in info:
        return str(info["img_path"])
    if isinstance(info.get("img_info"), dict) and "filename" in info["img_info"]:
        return str(info["img_info"]["filename"])
    return None


def subset_dataset_by_basenames(dataset: object, keep_basenames: Iterable[str]) -> object:
    infos, attr = _dataset_infos(dataset)
    keep = {str(name) for name in keep_basenames}
    filtered = []
    for info in infos:
        filename = _info_filename(info)
        if filename is None:
            continue
        if os.path.basename(filename) in keep:
            filtered.append(info)
    setattr(dataset, attr, filtered)
    return dataset


def sample_dataset_basenames(dataset: object, seed: int, limit: int) -> list[str]:
    infos, _ = _dataset_infos(dataset)
    basenames = []
    for info in infos:
        filename = _info_filename(info)
        if filename is None:
            continue
        basenames.append(os.path.basename(filename))
    basenames = sorted(set(basenames))
    if limit <= 0 or limit >= len(basenames):
        return basenames
    rng = random.Random(int(seed))
    return sorted(rng.sample(basenames, min(limit, len(basenames))))


def resolve_proxy_test_cfg(inference_mode: str) -> dict[str, Any]:
    if inference_mode == "whole":
        return AttrDict({"mode": "whole"})
    if inference_mode == "coarse_slide":
        return AttrDict({"mode": "slide", "stride": (448, 448), "crop_size": (448, 448)})
    raise ValueError(f"Unsupported proxy inference mode: {inference_mode}")


def build_validation_payload(
    *,
    eval_results: dict[str, Any],
    classes: list[str] | tuple[str, ...],
    validation_mode: str,
    used_inference_mode: str,
) -> dict[str, Any]:
    summary = {
        "mIoU": float(eval_results.get("mIoU", 0.0) * 100.0),
        "mAcc": float(eval_results.get("mAcc", 0.0) * 100.0),
        "aAcc": float(eval_results.get("aAcc", 0.0) * 100.0),
    }
    payload: dict[str, Any] = {
        "validation_mode": validation_mode,
        "used_inference_mode": used_inference_mode,
        "summary": summary,
        "proxy_summary": summary if validation_mode == "proxy" else None,
        "full_summary": summary if validation_mode == "full" else None,
    }
    per_class = {
        class_name: {
            "IoU": float(eval_results.get(f"IoU.{class_name}", 0.0) * 100.0),
            "Acc": float(eval_results.get(f"Acc.{class_name}", 0.0) * 100.0),
        }
        for class_name in classes
        if f"IoU.{class_name}" in eval_results or f"Acc.{class_name}" in eval_results
    }
    if per_class:
        payload["per_class"] = per_class
    return payload


_TABLE_ROW_RE = re.compile(r"^\|\s*(?P<class>.*?)\s*\|\s*(?P<iou>.*?)\s*\|\s*(?P<acc>.*?)\s*\|$")


def parse_per_class_from_log(log_text: str) -> dict[str, dict[str, float]]:
    lines = log_text.splitlines()
    last_table: dict[str, dict[str, float]] = {}
    in_table = False
    current: dict[str, dict[str, float]] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if "per class results:" in line:
            in_table = True
            current = {}
            continue
        if not in_table:
            continue
        if "Summary:" in line:
            if current:
                last_table = current
            in_table = False
            continue
        match = _TABLE_ROW_RE.match(line)
        if match is None:
            continue
        class_name = match.group("class").strip()
        if class_name in {"Class", ""}:
            continue
        try:
            iou = float(match.group("iou").strip())
            acc = float(match.group("acc").strip())
        except ValueError:
            continue
        current[class_name] = {"IoU": iou, "Acc": acc}

    if in_table and current:
        last_table = current
    return last_table


def _proxy_miou(candidate: dict[str, Any]) -> float:
    proxy_summary = candidate.get("proxy_summary") or {}
    return float(proxy_summary.get("mIoU", float("-inf")))


def _plan_length(candidate: dict[str, Any]) -> int:
    if "plan_length" in candidate:
        return int(candidate["plan_length"])
    transfer_pairs = candidate.get("transfer_pairs") or []
    return int(len(transfer_pairs))


def _mixture_distance(left: list[float], right: list[float]) -> float:
    return float(sum(abs(float(a) - float(b)) for a, b in zip(left, right)))


def select_full_eval_shortlist(candidates: list[dict[str, Any]], top_k: int = 3) -> list[str]:
    if top_k <= 0 or not candidates:
        return []
    ordered = sorted(candidates, key=_proxy_miou, reverse=True)
    top1 = ordered[0]
    selected: list[str] = [str(top1["candidate_id"])]

    if top_k == 1:
        return selected

    top_target = list(top1.get("target_mixture") or [])
    diversity_pool = [
        candidate
        for candidate in ordered[1:]
        if candidate.get("target_mixture") and _plan_length(candidate) < 3
    ]
    if not diversity_pool:
        diversity_pool = [candidate for candidate in ordered[1:] if candidate.get("target_mixture")]
    if diversity_pool:
        diverse = max(
            diversity_pool,
            key=lambda candidate: (
                _mixture_distance(top_target, list(candidate.get("target_mixture") or [])),
                _proxy_miou(candidate),
            ),
        )
        diverse_id = str(diverse["candidate_id"])
        if diverse_id not in selected:
            selected.append(diverse_id)

    if len(selected) >= top_k:
        return selected[:top_k]

    multistep_pool = [candidate for candidate in ordered if _plan_length(candidate) >= 3]
    if _plan_length(top1) >= 3:
        multistep_pool = [candidate for candidate in multistep_pool if str(candidate["candidate_id"]) != selected[0]]
    for candidate in multistep_pool:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id not in selected:
            selected.append(candidate_id)
            break

    if len(selected) >= top_k:
        return selected[:top_k]

    for candidate in ordered:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id not in selected:
            selected.append(candidate_id)
        if len(selected) >= top_k:
            break
    return selected[:top_k]


def is_cuda_oom_error(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda oom" in message or "cublas_status_alloc_failed" in message
