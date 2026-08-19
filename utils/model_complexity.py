"""Parameter and FLOP summaries for MV-SRBF training logs and paper tables."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from utils.input_config import resolve_padded_input_size


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def format_large_number(value: int | float) -> str:
    """Format large counts for concise console logs."""
    value = float(value)
    if abs(value) >= 1e9:
        return f"{value / 1e9:.3f}B"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.3f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.3f}K"
    return f"{value:.0f}"


def _tensor_output(output: Any) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    return None


def _resolve_aux_indices(cfg: Dict[str, Any], batch_size: int, device: torch.device) -> Dict[int, torch.Tensor]:
    encoder_cfg = cfg.get("model", {}).get("encoder", {})
    sie_cfg = encoder_cfg.get("sie", {})
    if not bool(sie_cfg.get("enabled", False)):
        return {}
    return {int(sie_cfg.get("dim_index", 0)): torch.zeros(batch_size, dtype=torch.long, device=device)}


def _dummy_full_modal_batch(
    model: nn.Module,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[List[str], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
    batch_size = 1
    height, width = resolve_padded_input_size(cfg)
    modal_keys: List[str] = list(getattr(model, "modal_keys", cfg.get("dataset", {}).get("modalities", {}).keys()))
    modalities = {
        key: torch.zeros(batch_size, 3, int(height), int(width), device=device)
        for key in modal_keys
    }
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    mask = torch.zeros(batch_size, len(modal_keys), dtype=torch.bool, device=device)
    aux_indices = _resolve_aux_indices(cfg, batch_size, device)
    return modal_keys, modalities, labels, mask, aux_indices


class _FullModalForward(nn.Module):
    """Tensor-only wrapper for fvcore tracing of MV-SRBF full-modal inference."""

    def __init__(
        self,
        model: nn.Module,
        modal_keys: List[str],
        labels: torch.Tensor,
        mask: torch.Tensor,
        aux_indices: Dict[int, torch.Tensor],
    ):
        super().__init__()
        self.model = model
        self.modal_keys = list(modal_keys)
        self.register_buffer("_labels", labels, persistent=False)
        self.register_buffer("_mask", mask, persistent=False)
        self.aux_dims = sorted(int(dim) for dim in aux_indices)
        for dim in self.aux_dims:
            self.register_buffer(f"_aux_{dim}", aux_indices[dim], persistent=False)

    def forward(self, *modal_tensors: torch.Tensor) -> torch.Tensor:
        modalities = {key: tensor for key, tensor in zip(self.modal_keys, modal_tensors)}
        aux_indices = {dim: getattr(self, f"_aux_{dim}") for dim in self.aux_dims}
        out = self.model(
            modalities,
            self._labels,
            mask=self._mask,
            aux_indices=aux_indices,
            return_loss_dict=False,
        )
        return out["fuse_logits"]


@torch.no_grad()
def _estimate_full_modal_flops_lightweight(model: nn.Module, cfg: Dict[str, Any], device: torch.device) -> int:
    """
    Estimate FLOPs for one full-modal inference forward with local hooks.

    The hook-based counter covers Conv2d and Linear modules, which dominate CNN,
    ViT patch embedding, MLP, head, and projection costs. Unsupported functional
    operators are ignored, so the value is an estimate for monitoring and config
    comparison rather than a formal benchmark number.
    """
    flops = 0
    handles = []

    def conv_hook(module: nn.Conv2d, inputs: Tuple[Any, ...], output: Any) -> None:
        nonlocal flops
        out = _tensor_output(output)
        if out is None or out.ndim < 4:
            return
        batch, out_channels, out_h, out_w = out.shape[:4]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        flops += int(batch * out_channels * out_h * out_w * kernel_ops)

    def linear_hook(module: nn.Linear, inputs: Tuple[Any, ...], output: Any) -> None:
        nonlocal flops
        out = _tensor_output(output)
        inp = inputs[0] if inputs else None
        if out is None or not torch.is_tensor(inp):
            return
        flops += int(out.numel() * module.in_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    _, modalities, labels, mask, aux_indices = _dummy_full_modal_batch(model, cfg, device)

    try:
        model(modalities, labels, mask=mask, aux_indices=aux_indices, return_loss_dict=False)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()
    return int(flops)


@torch.no_grad()
def _estimate_full_modal_flops_fvcore(model: nn.Module, cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Estimate full-modal FLOPs with fvcore's FlopCountAnalysis."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as exc:
        raise ImportError("fvcore is required for publication-table FLOPs. Install with `pip install fvcore`.") from exc

    was_training = model.training
    model.eval()
    modal_keys, modalities, labels, mask, aux_indices = _dummy_full_modal_batch(model, cfg, device)
    wrapper = _FullModalForward(model, modal_keys, labels, mask, aux_indices).to(device)
    inputs = tuple(modalities[key] for key in modal_keys)
    try:
        analysis = FlopCountAnalysis(wrapper, inputs)
        total = int(analysis.total())
        unsupported = dict(Counter({str(k): int(v) for k, v in analysis.unsupported_ops().items()}))
        uncalled = sorted(str(name) for name in analysis.uncalled_modules())
    finally:
        if was_training:
            model.train()
    return {
        "value": total,
        "backend": "fvcore",
        "unsupported_ops": unsupported,
        "uncalled_modules": uncalled,
    }


def profile_full_modal_flops(
    model: nn.Module,
    cfg: Dict[str, Any],
    device: torch.device,
    backend: str = "fvcore",
    allow_fallback: bool = True,
) -> Dict[str, Any]:
    """Profile one full-modal inference forward, preferring fvcore for publication tables."""
    backend = str(backend or "fvcore").lower()
    if backend not in {"fvcore", "lightweight"}:
        raise ValueError(f"Unknown FLOPs backend: {backend!r}. Use 'fvcore' or 'lightweight'.")
    if backend == "lightweight":
        return {
            "value": _estimate_full_modal_flops_lightweight(model, cfg, device),
            "backend": "lightweight",
            "unsupported_ops": {},
            "uncalled_modules": [],
        }
    try:
        return _estimate_full_modal_flops_fvcore(model, cfg, device)
    except Exception as exc:
        if not allow_fallback:
            raise
        return {
            "value": _estimate_full_modal_flops_lightweight(model, cfg, device),
            "backend": "lightweight_fallback",
            "fallback_reason": str(exc),
            "unsupported_ops": {},
            "uncalled_modules": [],
        }


def summarize_model_complexity(model: nn.Module, cfg: Dict[str, Any], device: torch.device) -> Dict[str, float]:
    """Build numeric and formatted model complexity fields for logs."""
    params = count_parameters(model)
    flops = profile_full_modal_flops(model, cfg, device)["value"]
    return {
        "params_total": float(params["total"]),
        "params_trainable": float(params["trainable"]),
        "flops_full_modal": float(flops),
    }
