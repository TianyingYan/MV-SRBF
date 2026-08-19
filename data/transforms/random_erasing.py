"""Batch-level random erasing for multi-modal tensors."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torchvision.transforms import RandomErasing
from torchvision.transforms import functional as F
import torchvision.transforms.v2 as v2


_SYNC_ALIASES = {"synchronous", "sync", "shared"}
_ASYNC_ALIASES = {"asynchronous", "async", "independent"}


def _resolve_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value in _SYNC_ALIASES:
        return "synchronous"
    if value in _ASYNC_ALIASES:
        return "asynchronous"
    raise ValueError(
        f"Unknown random_erasing.mode {mode!r}. "
        "Use 'synchronous' (same erase region on all modalities) or "
        "'asynchronous' (independent erase region per modality)."
    )


def apply_random_erasing_batch(
    modalities: Dict[str, torch.Tensor],
    *,
    p: float = 0.5,
    scale: tuple = (0.02, 0.33),
    ratio: tuple = (0.3, 3.3),
    mode: str = "asynchronous",
) -> Dict[str, torch.Tensor]:
    """Apply random erasing to each modality tensor shaped [B,C,H,W]."""
    out: Dict[str, torch.Tensor] = {}
    mode = _resolve_mode(mode)
    transform = v2.RandomErasing(p=1.0, scale=scale, ratio=ratio, value=0.0)
    if mode == "synchronous":
        keys = list(modalities.keys())
        if not keys:
            return modalities
        batch_size = modalities[keys[0]].shape[0]
        device = modalities[keys[0]].device
        apply_mask = torch.rand(batch_size, device=device) < p
        out = {key: value.clone() for key, value in modalities.items()}
        for idx in range(batch_size):
            if not bool(apply_mask[idx].item()):
                continue
            reference = modalities[keys[0]][idx]
            i, j, h, w, value = RandomErasing.get_params(
                reference,
                scale=scale,
                ratio=ratio,
                value=[0.0],
            )
            for key in keys:
                sample = out[key][idx]
                erase_value = value.to(device=sample.device, dtype=sample.dtype)
                out[key][idx] = F.erase(sample, i, j, h, w, erase_value, inplace=False)
        return out

    for key, x in modalities.items():
        y = x.clone()
        for idx in range(x.shape[0]):
            if torch.rand(1, device=x.device).item() < p:
                y[idx : idx + 1] = transform(x[idx : idx + 1])
        out[key] = y
    return out


class RandomErasingBatchAugment:
    """Batch-level random erasing for multi-modal tensors, used by the trainer.

    Configured via the ``input.train_augmentation.batch_augment.random_erasing`` block::

        random_erasing:
          enable: true
          mode: "asynchronous"   # or "synchronous"
          p: 0.5
          scale: [0.02, 0.4]
          ratio: [0.3, 3.33]

    Two modes select how erasing is shared across modalities:
      - ``synchronous``:  the SAME erase rectangle (and value) is applied to every
        modality of a sample, preserving cross-modal spatial alignment.
      - ``asynchronous``: each modality is erased independently (a different region per
        modality), encouraging modality-specific robustness.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        re_cfg = self.cfg.get("random_erasing", {})
        if not isinstance(re_cfg, dict):
            raise ValueError("input.train_augmentation.batch_augment.random_erasing must be a mapping.")

        self.enable_re = bool(re_cfg.get("enable", True))
        self.mode = _resolve_mode(re_cfg.get("mode", "asynchronous"))
        self.re_p = float(re_cfg.get("p", 0.5))
        self.re_scale = tuple(re_cfg.get("scale", (0.02, 0.33)))
        self.re_ratio = tuple(re_cfg.get("ratio", (0.3, 3.3)))

    def __call__(self, modalities: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.enable_re:
            return modalities
        return apply_random_erasing_batch(
            modalities,
            p=self.re_p,
            scale=self.re_scale,
            ratio=self.re_ratio,
            mode=self.mode,
        )
