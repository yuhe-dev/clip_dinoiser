from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch.nn as nn
import torch


DEFAULT_TRAINABLE_MODULES = [
    "obj_proj",
    "bkg_decoder",
]


@dataclass
class TrainabilitySummary:
    trainable_modules: List[str]
    trainable_param_names: List[str]
    found_param_names: List[str]
    corr_param_names: List[str]
    backbone_param_names: List[str]


@dataclass
class ModuleGradSummary:
    module_path: str
    param_count: int
    params_with_grad: int
    total_grad_norm: float


def normalize_module_paths(module_paths: Iterable[str] | None) -> List[str]:
    if not module_paths:
        return list(DEFAULT_TRAINABLE_MODULES)
    ordered: List[str] = []
    seen: set[str] = set()
    for item in module_paths:
        value = str(item).strip()
        if not value or value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered or list(DEFAULT_TRAINABLE_MODULES)


def resolve_module(root: nn.Module, module_path: str) -> nn.Module:
    current: object = root
    for token in str(module_path).split("."):
        if token.lstrip("-").isdigit():
            index = int(token)
            current = current[index]  # type: ignore[index]
            continue
        if not hasattr(current, token):
            raise AttributeError(f"Module path '{module_path}' could not resolve token '{token}'")
        current = getattr(current, token)
    if not isinstance(current, nn.Module):
        raise TypeError(f"Resolved object for '{module_path}' is not an nn.Module")
    return current


def configure_trainable_modules(model: nn.Module, module_paths: Iterable[str] | None) -> List[str]:
    normalized = normalize_module_paths(module_paths)
    for param in model.parameters():
        param.requires_grad = False
    for path in normalized:
        module = resolve_module(model, path)
        for param in module.parameters():
            param.requires_grad = True
    return normalized


def set_train_mode_for_modules(model: nn.Module, module_paths: Iterable[str] | None) -> List[str]:
    normalized = normalize_module_paths(module_paths)
    model.eval()
    for path in normalized:
        resolve_module(model, path).train()
    return normalized


def build_optimizer_groups(
    model: nn.Module,
    *,
    corr_lr: float,
    found_lr: float,
    backbone_lr: float | None = None,
) -> tuple[list[dict], TrainabilitySummary]:
    backbone_lr = float(backbone_lr if backbone_lr is not None else corr_lr)
    found_params: list = []
    corr_params: list = []
    backbone_params: list = []
    found_param_names: List[str] = []
    corr_param_names: List[str] = []
    backbone_param_names: List[str] = []
    trainable_param_names: List[str] = []

    for name, param in model.named_parameters():
        if not bool(param.requires_grad):
            continue
        trainable_param_names.append(name)
        if name.startswith("bkg_decoder."):
            found_params.append(param)
            found_param_names.append(name)
        elif name.startswith("clip_backbone.backbone."):
            backbone_params.append(param)
            backbone_param_names.append(name)
        else:
            corr_params.append(param)
            corr_param_names.append(name)

    groups: list[dict] = []
    if corr_params:
        groups.append({"params": corr_params, "lr": float(corr_lr)})
    if found_params:
        groups.append({"params": found_params, "lr": float(found_lr)})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": float(backbone_lr)})

    summary = TrainabilitySummary(
        trainable_modules=[],
        trainable_param_names=trainable_param_names,
        found_param_names=found_param_names,
        corr_param_names=corr_param_names,
        backbone_param_names=backbone_param_names,
    )
    return groups, summary


def collect_module_grad_summaries(model: nn.Module, module_paths: Iterable[str] | None) -> List[ModuleGradSummary]:
    summaries: List[ModuleGradSummary] = []
    for module_path in normalize_module_paths(module_paths):
        module = resolve_module(model, module_path)
        params = list(module.parameters())
        total_sq = 0.0
        params_with_grad = 0
        for param in params:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            norm = float(torch.linalg.vector_norm(grad).item())
            total_sq += norm * norm
            params_with_grad += 1
        summaries.append(
            ModuleGradSummary(
                module_path=str(module_path),
                param_count=len(params),
                params_with_grad=params_with_grad,
                total_grad_norm=total_sq ** 0.5,
            )
        )
    return summaries
