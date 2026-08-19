"""Resolve adaptive input settings shared by data and model builders."""
from __future__ import annotations

import re
from typing import Any, Dict, Tuple


VEHICLE_SIZE = (128, 256)


def _norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def resolve_input_size(cfg: Dict[str, Any]) -> Tuple[int, int]:
    """Resolve input.size; 'auto' uses the vehicle ReID size [128, 256]."""
    inp = cfg.get("input", {})
    blocked_size_key = "task" + "_size"
    if blocked_size_key in inp:
        raise ValueError("Unsupported input size table. Use input.size: 'auto' or [height, width].")
    size = inp.get("size", "auto")
    if isinstance(size, str):
        if size.lower() != "auto":
            raise ValueError("input.size must be [height, width] or 'auto'.")
        return VEHICLE_SIZE
    return int(size[0]), int(size[1])


def resolve_input_padding(cfg: Dict[str, Any]) -> Tuple[int, int]:
    """Resolve global input padding as symmetric (height, width) pixels."""
    inp = cfg.get("input", {})
    padding = inp.get("padding", (0, 0))
    if padding is None:
        return 0, 0
    if isinstance(padding, (int, float)):
        pad_h = pad_w = int(padding)
    else:
        if len(padding) != 2:
            raise ValueError("input.padding must be an int or [height, width].")
        pad_h, pad_w = int(padding[0]), int(padding[1])
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"input.padding values must be non-negative, got {(pad_h, pad_w)}.")
    return pad_h, pad_w


def resolve_padded_input_size(cfg: Dict[str, Any]) -> Tuple[int, int]:
    """Return the actual tensor size produced by resize, padding, and optional crop."""
    h, w = resolve_input_size(cfg)
    shared_aug = cfg.get("input", {}).get("train_augmentation", {}).get("shared", {})
    if shared_aug.get("random_crop"):
        crop_cfg = shared_aug.get("random_crop")
        if isinstance(crop_cfg, dict) and crop_cfg.get("size") is not None:
            size = crop_cfg["size"]
            return int(size[0]), int(size[1])
        return h, w
    pad_h, pad_w = resolve_input_padding(cfg)
    return h + 2 * pad_h, w + 2 * pad_w


def resolve_resize_interpolation(cfg: Dict[str, Any]) -> str:
    """Resolve interpolation; 'auto' uses bicubic for transformer backbones and bilinear otherwise."""
    inp = cfg.get("input", {})
    interp = str(inp.get("resize_interpolation", "auto")).lower()
    if interp != "auto":
        return interp
    name = _norm_name(
        str(cfg.get("model", {}).get("encoder", {}).get("backbone", {}).get("name", ""))
    )
    if any(token in name for token in ("vit", "swin", "maxvit", "transformer")):
        return "bicubic"
    return "bilinear"
