"""Build torch.optim optimizers from merged YAML config (solver.optimizer)."""
from __future__ import annotations

import inspect
from typing import Any, Dict, List

import torch
import torch.nn as nn


def _filter_kwargs_for_callable(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    params = sig.parameters
    out: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in params:
            out[k] = v
    return out


def build_optimizer(cfg: Dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    """
    Resolve solver.optimizer.name to torch.optim.<Name> and instantiate with
    model.parameters() plus constructor kwargs from cfg (filtered per signature).
    """
    sol = cfg.get("solver", {})
    opt_cfg = sol.get("optimizer", {})
    removed_fields = sorted(set(opt_cfg).intersection({"backbone_lr", "non_backbone_lr", "backbone_prefixes", "module_prefixes"}))
    if removed_fields:
        raise ValueError(
            "Removed optimizer field(s): "
            + ", ".join(removed_fields)
            + ". Use solver.optimizer.module_lrs with the built-in module groups."
        )
    name = opt_cfg.get("name")
    if not name or not isinstance(name, str):
        raise ValueError("solver.optimizer.name must be a non-empty string")

    cls = getattr(torch.optim, name, None)
    if cls is None or not (isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer)):
        raise ValueError(f"Unknown torch.optim optimizer: {name!r}")

    # torch.optim uses `lr`; YAML may use base_lr (BoT style)
    lr = float(opt_cfg.get("lr", opt_cfg.get("base_lr", 3e-4)))
    wd = float(opt_cfg.get("weight_decay", 0.0))
    momentum = float(opt_cfg.get("momentum", 0.9))
    nesterov = bool(opt_cfg.get("nesterov", False))
    betas = opt_cfg.get("betas", (0.9, 0.999))
    if isinstance(betas, list):
        betas = tuple(betas)

    raw_kwargs: Dict[str, Any] = {
        "lr": lr,
        "weight_decay": wd,
        "momentum": momentum,
        "nesterov": nesterov,
        "betas": betas,
    }
    if "eps" in opt_cfg:
        raw_kwargs["eps"] = float(opt_cfg["eps"])
    if "kwargs" in opt_cfg and isinstance(opt_cfg["kwargs"], dict):
        raw_kwargs.update(opt_cfg["kwargs"])

    ctor_kwargs = _filter_kwargs_for_callable(cls.__init__, raw_kwargs)

    use_param_groups = bool(opt_cfg.get("use_param_groups", False))
    module_lrs = opt_cfg.get("module_lrs", {}) or {}
    if module_lrs and not use_param_groups:
        raise ValueError("solver.optimizer.module_lrs requires use_param_groups: true.")

    if use_param_groups:
        bias_lr_factor = float(opt_cfg.get("bias_lr_factor", 2.0))
        wd_bias = float(opt_cfg.get("weight_decay_bias", 0.0))
        if module_lrs:
            groups = param_groups_with_module_lrs(
                model,
                default_lr=lr,
                module_lrs=module_lrs,
                bias_lr_factor=bias_lr_factor,
                weight_decay=wd,
                weight_decay_bias=wd_bias,
            )
        else:
            groups = param_groups_with_bias_lr(model, lr, bias_lr_factor, wd, wd_bias)
        group_ctor = {k: v for k, v in ctor_kwargs.items() if k not in ("lr", "weight_decay")}
        return cls(groups, **group_ctor)

    return cls(model.parameters(), **ctor_kwargs)


def _default_group_prefixes() -> Dict[str, tuple[str, ...]]:
    return {
        "backbone": ("encoder.trunk.",),
        "encoder": ("encoder.sie",),
        "modal_heads": ("modal_heads.", "local_global_reducers."),
        "recovery": ("mask_token", "recovery."),
        "fusion": ("modal_mlp.", "fuse_mlp.", "fuse_head."),
    }


def _module_name_for_param(param_name: str, prefixes: Dict[str, tuple[str, ...]]) -> str:
    if param_name.startswith("encoder.encoders."):
        if ".trunk." in param_name:
            return "backbone"
        if ".sie." in param_name:
            return "encoder"
    best_name = "other"
    best_len = -1
    for module_name, candidates in prefixes.items():
        for prefix in candidates:
            if param_name.startswith(prefix) and len(prefix) > best_len:
                best_name = module_name
                best_len = len(prefix)
    return best_name


def param_groups_with_module_lrs(
    model: nn.Module,
    default_lr: float,
    module_lrs: Dict[str, Any],
    bias_lr_factor: float,
    weight_decay: float,
    weight_decay_bias: float,
) -> List[Dict[str, Any]]:
    """Build named module groups with independent base learning rates."""
    prefixes = _default_group_prefixes()
    resolved_lrs = {str(name): float(value) for name, value in module_lrs.items()}
    allowed = set(prefixes) | {"other"}
    unknown = sorted(set(resolved_lrs) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown solver.optimizer.module_lrs group(s): {unknown}. "
            f"Use one or more of {sorted(allowed)}."
        )

    module_by_param_id: Dict[int, nn.Module] = {}
    for module in model.modules():
        for param in module.parameters(recurse=False):
            module_by_param_id[id(param)] = module

    buckets: Dict[tuple[str, bool], List[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        module_name = _module_name_for_param(name, prefixes)
        no_decay = _is_no_decay_param(name, module_by_param_id.get(id(param), model))
        buckets.setdefault((module_name, no_decay), []).append(param)

    out: List[Dict[str, Any]] = []
    for (module_name, no_decay), params in buckets.items():
        group_lr = resolved_lrs.get(module_name, resolved_lrs.get("other", default_lr))
        if no_decay:
            group_lr *= bias_lr_factor
        out.append(
            {
                "params": params,
                "lr": group_lr,
                "base_lr": group_lr,
                "weight_decay": weight_decay_bias if no_decay else weight_decay,
                "module_name": module_name,
                "no_decay": no_decay,
                "is_backbone": module_name == "backbone",
            }
        )
    return out


def param_groups_with_bias_lr(
    model: nn.Module,
    base_lr: float,
    bias_lr_factor: float,
    weight_decay: float,
    weight_decay_bias: float,
) -> List[Dict[str, Any]]:
    """Optional BoT-style param groups (higher lr for bias, no wd on bias/BN)."""
    params_decay: List[nn.Parameter] = []
    params_no_decay: List[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            for p in m.parameters(recurse=False):
                if p.requires_grad:
                    params_no_decay.append(p)
        else:
            for n, p in m.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                if n.endswith("bias"):
                    params_no_decay.append(p)
                else:
                    params_decay.append(p)
    # Fallback: collect any missed parameters
    covered = set(id(p) for p in params_decay + params_no_decay)
    for p in model.parameters():
        if p.requires_grad and id(p) not in covered:
            params_decay.append(p)

    return [
        {"params": params_decay, "lr": base_lr, "base_lr": base_lr, "weight_decay": weight_decay, "module_name": "all"},
        {
            "params": params_no_decay,
            "lr": base_lr * bias_lr_factor,
            "base_lr": base_lr * bias_lr_factor,
            "weight_decay": weight_decay_bias,
            "module_name": "all",
            "no_decay": True,
        },
    ]


def _is_no_decay_param(name: str, module: nn.Module) -> bool:
    """Match common ReID optimizer rules: no weight decay for bias and norm parameters."""
    if name.endswith(".bias"):
        return True
    return isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.InstanceNorm1d, nn.InstanceNorm2d))
