import math
import inspect
from typing import Any, Dict, Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupMultiStepLR(_LRScheduler):
    """Multi-step learning-rate schedule with linear warmup."""

    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=0.1, warmup_updates=500, last_epoch=-1):
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_updates = warmup_updates
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_updates:
            alpha = self.last_epoch / self.warmup_updates
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return [
            base_lr * self.gamma ** sum(self.last_epoch >= milestone for milestone in self.milestones)
            for base_lr in self.base_lrs
        ]


class CosineAnnealingWarmupLR(_LRScheduler):
    """Cosine learning-rate schedule with linear warmup."""

    def __init__(self, optimizer, T_max, eta_min=0, warmup_factor=0.1, warmup_updates=500, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_factor = warmup_factor
        self.warmup_updates = warmup_updates
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_updates:
            alpha = self.last_epoch / self.warmup_updates
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        denom = max(1, self.T_max - self.warmup_updates)
        progress = (self.last_epoch - self.warmup_updates) / denom
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
            for base_lr in self.base_lrs
        ]


def _filter_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def build_lr_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer, total_steps: Optional[int] = None):
    """Build a configured LR scheduler from local warmup schedules or torch built-ins."""
    sol = cfg.get("solver", {})
    sch_cfg = sol.get("lr_scheduler", {})
    name = str(sch_cfg.get("name", "CosineAnnealingWarmupLR"))
    max_epochs = int(sol.get("max_epochs", 120))
    schedule_steps = int(total_steps) if total_steps is not None else int(sch_cfg.get("T_max", max_epochs))
    warmup_updates = max(0, int(sol.get("warmup_updates", sch_cfg.get("warmup_updates", 0))))

    if name == "CosineAnnealingWarmupLR":
        return CosineAnnealingWarmupLR(
            optimizer,
            T_max=int(sch_cfg.get("T_max", schedule_steps)),
            eta_min=float(sch_cfg.get("eta_min", 0.0)),
            warmup_factor=float(sch_cfg.get("warmup_factor", 0.1)),
            warmup_updates=warmup_updates,
            last_epoch=int(sch_cfg.get("last_epoch", -1)),
        )
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            milestones=list(sch_cfg.get("milestones", [])),
            gamma=float(sch_cfg.get("gamma", 0.1)),
            warmup_factor=float(sch_cfg.get("warmup_factor", 0.1)),
            warmup_updates=warmup_updates,
            last_epoch=int(sch_cfg.get("last_epoch", -1)),
        )

    cls = getattr(torch.optim.lr_scheduler, name, None)
    if cls is None:
        raise ValueError(f"Unknown LR scheduler: {name!r}")
    raw_kwargs = dict(sch_cfg.get("kwargs", {}))
    raw_kwargs.update({k: v for k, v in sch_cfg.items() if k not in {"name", "kwargs"}})
    raw_kwargs.setdefault("T_max", schedule_steps)
    raw_kwargs.setdefault("step_size", max(1, schedule_steps // 3))
    raw_kwargs.setdefault("milestones", sch_cfg.get("milestones", []))
    raw_kwargs.setdefault("gamma", sch_cfg.get("gamma", 0.1))
    return cls(optimizer, **_filter_kwargs(cls.__init__, raw_kwargs))
