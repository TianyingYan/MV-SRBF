"""Grad-CAM helpers for encoder and fusion classification stages."""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _active_encoder(model: nn.Module, modal_key: Optional[str] = None) -> nn.Module:
    encoder = getattr(model, "encoder", None)
    encoders = getattr(encoder, "encoders", None)
    if encoders is not None and len(encoders) > 0:
        keys = list(getattr(model, "modal_keys", []))
        index = keys.index(modal_key) if modal_key in keys else 0
        return encoders[index]
    return encoder


def _block_pre_attention_norm(block: nn.Module) -> Optional[nn.Module]:
    for name in ("ln_1", "norm1", "norm_1", "layernorm_before"):
        layer = getattr(block, name, None)
        if isinstance(layer, nn.Module):
            return layer
    return None


def resolve_grad_cam_target_layer(
    model: nn.Module, target_layer: str = "auto", modal_key: Optional[str] = None
) -> nn.Module:
    """Resolve the active modality's last spatial/patch-sensitive CAM layer."""
    encoder = _active_encoder(model, modal_key)
    if target_layer and target_layer != "auto":
        modules = dict(model.named_modules())
        if target_layer in modules:
            return modules[target_layer]
        local_modules = dict(encoder.named_modules()) if encoder is not None else {}
        if target_layer in local_modules:
            return local_modules[target_layer]
        raise ValueError(f"Unknown Grad-CAM target layer {target_layer!r}.")
    trunk = getattr(encoder, "trunk", None)
    if trunk is None:
        raise ValueError("Grad-CAM auto target requires model.encoder.trunk.")

    layers = getattr(getattr(trunk, "encoder", None), "layers", None)
    if layers is not None and len(layers) > 0:
        norm = _block_pre_attention_norm(layers[-1])
        if norm is not None:
            return norm

    transformer = getattr(trunk, "transformer", None)
    resblocks = getattr(transformer, "resblocks", None)
    if resblocks is not None and len(resblocks) > 0:
        norm = _block_pre_attention_norm(resblocks[-1])
        if norm is not None:
            return norm

    blocks = getattr(trunk, "blocks", None)
    if blocks is not None and len(blocks) > 0:
        norm = _block_pre_attention_norm(blocks[-1])
        if norm is not None:
            return norm

    hf_layers = getattr(getattr(trunk, "encoder", None), "layer", None)
    if hf_layers is not None and len(hf_layers) > 0:
        norm = _block_pre_attention_norm(hf_layers[-1])
        if norm is not None:
            return norm

    trunk_name = type(trunk).__name__.lower()
    if any(token in trunk_name for token in ("vit", "swin", "beit", "deit", "transformer")):
        norms = [module for module in trunk.modules() if isinstance(module, nn.LayerNorm)]
        if norms:
            return norms[-1]

    last_conv = None
    for module in trunk.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("Could not infer a Grad-CAM target layer. Pass --target_layer explicitly.")
    return last_conv


def infer_token_hw(model: nn.Module, x: torch.Tensor, modal_key: Optional[str] = None) -> Optional[Tuple[int, int]]:
    """Infer ViT patch grid size from the active patch embedding convolution."""
    encoder = _active_encoder(model, modal_key)
    trunk = getattr(encoder, "trunk", None)
    conv = getattr(trunk, "conv_proj", None)
    if conv is None:
        conv = getattr(trunk, "conv1", None)
    if conv is None:
        conv = getattr(getattr(trunk, "patch_embed", None), "proj", None)
    if conv is None:
        patch_embeddings = getattr(getattr(trunk, "embeddings", None), "patch_embeddings", None)
        conv = getattr(patch_embeddings, "projection", None)
    if not isinstance(conv, nn.Conv2d):
        return None
    with torch.no_grad():
        out = conv(x[:1])
    return int(out.shape[-2]), int(out.shape[-1])


def build_token_reshape_transform(spatial_hw: Optional[Tuple[int, int]]):
    """Return a reshape_transform for ViT token activations, or None for CNNs."""
    if spatial_hw is None:
        return None
    h, w = spatial_hw

    def factor_grid(num_tokens: int) -> Tuple[int, int, float]:
        ratio = h / max(w, 1)
        candidates = []
        for height in range(1, int(math.sqrt(num_tokens)) + 1):
            if num_tokens % height:
                continue
            width = num_tokens // height
            for hh, ww in ((height, width), (width, height)):
                score = abs(math.log(max(hh / max(ww, 1), 1e-8) / max(ratio, 1e-8)))
                candidates.append((score, hh, ww))
        if not candidates:
            raise ValueError(f"Cannot factor {num_tokens} tokens into a spatial grid.")
        score, hh, ww = min(candidates)
        return hh, ww, score

    def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]
        if tensor.ndim == 4:
            expected = h / max(w, 1)
            bchw_ratio = tensor.shape[2] / max(tensor.shape[3], 1)
            bhwc_ratio = tensor.shape[1] / max(tensor.shape[2], 1)
            bchw_score = abs(math.log(max(bchw_ratio, 1e-8) / max(expected, 1e-8)))
            bhwc_score = abs(math.log(max(bhwc_ratio, 1e-8) / max(expected, 1e-8)))
            return tensor.permute(0, 3, 1, 2) if bhwc_score < bchw_score else tensor
        if tensor.ndim != 3:
            return tensor
        expected_counts = {h * w, h * w + 1}
        if tensor.shape[0] in expected_counts or (tensor.shape[1] <= 8 and tensor.shape[0] > tensor.shape[1]):
            tensor = tensor.permute(1, 0, 2)
        if tensor.shape[1] == h * w + 1:
            tensor = tensor[:, 1:, :]
        token_count = tensor.shape[1]
        if token_count == h * w:
            out_h, out_w = h, w
        else:
            no_cls_h, no_cls_w, no_cls_score = factor_grid(token_count)
            cls_choice = None
            if token_count > 1:
                cls_h, cls_w, cls_score = factor_grid(token_count - 1)
                cls_choice = (cls_score + 0.05, cls_h, cls_w)
            if cls_choice is not None and cls_choice[0] < no_cls_score:
                tensor = tensor[:, 1:, :]
                out_h, out_w = cls_choice[1], cls_choice[2]
            else:
                out_h, out_w = no_cls_h, no_cls_w
        return tensor.reshape(tensor.shape[0], out_h, out_w, tensor.shape[-1]).permute(0, 3, 1, 2)

    return reshape_transform


class MVSRBFGradCAMWrapper(nn.Module):
    """Expose one encoder or fusion classifier for pytorch-grad-cam."""

    def __init__(
        self,
        model: nn.Module,
        *,
        modal_key: str,
        fixed_modalities: Dict[str, torch.Tensor],
        aux_indices: Optional[Dict[int, torch.Tensor]] = None,
        stage: str = "fusion",
    ):
        super().__init__()
        if stage not in {"encoder", "fusion"}:
            raise ValueError("Grad-CAM stage must be 'encoder' or 'fusion'.")
        self.model = model
        self.modal_key = modal_key
        self.fixed_modalities = {k: v.detach() for k, v in fixed_modalities.items()}
        self.aux_indices = aux_indices or {}
        self.stage = stage
        self.modal_keys = list(model.modal_keys)
        if modal_key not in self.modal_keys:
            raise KeyError(f"Unknown modality {modal_key!r}; available: {self.modal_keys}")
        self.modal_index = self.modal_keys.index(modal_key)

    def _encode(self, x: torch.Tensor, modal_key: str) -> torch.Tensor:
        modality_id = self.modal_keys.index(modal_key)
        modality_ids = torch.full((x.shape[0],), modality_id, dtype=torch.long, device=x.device)
        aux = {int(k): v.to(x.device) for k, v in self.aux_indices.items()}
        return self.model.encoder(x, modality_ids=modality_ids, aux_indices=aux).cls_pre

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_cls = self._encode(x, self.modal_key)
        if self.stage == "encoder":
            modal_mlp = getattr(self.model, "modal_mlp", None)
            modal_feat = modal_mlp(target_cls) if modal_mlp is not None else target_cls
            logits, _, _ = self.model.modal_heads[self.modal_key](modal_feat)
            return logits

        if type(self.model).forward is not nn.Module.forward:
            modalities = {
                key: (x if key == self.modal_key else self.fixed_modalities[key].to(x.device))
                for key in self.modal_keys
            }
            labels = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            mask = torch.zeros(x.shape[0], len(self.modal_keys), dtype=torch.bool, device=x.device)
            return self.model(
                modalities, labels, mask=mask,
                aux_indices={int(k): v.to(x.device) for k, v in self.aux_indices.items()},
                return_loss_dict=False,
            )["fuse_logits"]

        cls_list = []
        for key in self.modal_keys:
            if key == self.modal_key:
                cls_list.append(target_cls)
            else:
                fixed = self.fixed_modalities[key].to(x.device)
                with torch.no_grad():
                    cls_list.append(self._encode(fixed, key).detach())
        cls_stack = torch.stack(cls_list, dim=1)
        fuse_pre = self.model.fuse_mlp(cls_stack.flatten(1))
        logits, _, _ = self.model.fuse_head(fuse_pre)
        return logits


def tensor_to_rgb_float(chw: torch.Tensor, mean=None, std=None) -> np.ndarray:
    """Convert a CHW tensor to float RGB in [0, 1] for CAM overlays."""
    x = chw.detach().cpu().float().clone()
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
        std_t = torch.tensor(std, dtype=x.dtype).view(-1, 1, 1)
        x = x * std_t + mean_t
    x = x.clamp(0, 1)
    return x.permute(1, 2, 0).numpy()
