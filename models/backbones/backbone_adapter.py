"""Backbone bridge for torchvision, torch hub, and custom image models."""
from __future__ import annotations

import importlib
import inspect
import math
import re
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import BackboneOutput


FAMILY_ALIASES = {
    "alexnet": "alexnet",
    "convnext": "convnext_tiny",
    "densenet": "densenet121",
    "efficientnet": "efficientnet_b0",
    "efficientnetv2": "efficientnet_v2_s",
    "googlenet": "googlenet",
    "inceptionv3": "inception_v3",
    "maxvit": "maxvit_t",
    "mnasnet": "mnasnet1_0",
    "mobilenetv2": "mobilenet_v2",
    "mobilenetv3": "mobilenet_v3_large",
    "regnet": "regnet_y_400mf",
    "resnet": "resnet50",
    "resnext": "resnext50_32x4d",
    "shufflenetv2": "shufflenet_v2_x1_0",
    "squeezenet": "squeezenet1_0",
    "swintransformer": "swin_t",
    "vgg": "vgg16",
    "visiontransformer": "vit_b_16",
    "wideresnet": "wide_resnet50_2",
}


def normalize_model_name(name: str) -> str:
    """Normalize human-readable backbone names to torchvision factory style."""
    key = re.sub(r"[^a-z0-9]+", "", name.lower())
    if key in FAMILY_ALIASES:
        return FAMILY_ALIASES[key]
    return name.lower().replace(" ", "_").replace("-", "_")


def import_object(target: str) -> Any:
    """Import an object from 'module.submodule:object_name'."""
    if ":" not in target:
        raise ValueError("Custom backbone target must use 'module:object' syntax.")
    module_name, object_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _url_suffix(source: str) -> str:
    return Path(urlparse(source).path).suffix.lower()


def _download_safetensors_url(source: str) -> Dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "Loading .safetensors weights requires `safetensors`. "
            "Install it with `pip install safetensors` or use a .bin/.pth file."
        ) from exc
    filename = Path(urlparse(source).path).name or "downloaded_model.safetensors"
    cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / filename
    if not cached_file.is_file():
        torch.hub.download_url_to_file(source, str(cached_file), progress=True)
    return load_file(str(cached_file), device="cpu")


def _load_state_dict_source(source: str) -> Dict[str, torch.Tensor]:
    """Load a state dict from torch, safetensors, or torch model_zoo sources."""
    source = str(source)
    if _is_url(source):
        try:
            print(f"Loading pretrained weights from URL: {source}")
            if _url_suffix(source) == ".safetensors":
                state = _download_safetensors_url(source)
            else:
                state = torch.utils.model_zoo.load_url(source, map_location="cpu")
            print(f"Downloaded pretrained weights from URL: {source}")
        except Exception as exc:
            raise RuntimeError(
                "Failed to download pretrained weights from URL. "
                f"URL: {source}. Check network access, proxy settings, or provide "
                "a local model.encoder.backbone.pretrained_weights_path for offline runs."
            ) from exc
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(
                f"Pretrained weight file not found: {path}. "
                "Check model.encoder.backbone.pretrained_weights_path or disable explicit weight loading."
            )
        print(f"Loading pretrained weights from local file: {path}")
        if path.suffix.lower() == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError(
                    "Loading .safetensors weights requires `safetensors`. "
                    "Install it with `pip install safetensors` or use a .bin/.pth file."
                ) from exc
            state = load_file(str(path), device="cpu")
        else:
            state = None
            if path.suffix.lower() == ".pt":
                try:
                    state = torch.jit.load(str(path), map_location="cpu")
                except RuntimeError:
                    state = None
            try:
                if state is None:
                    state = torch.load(str(path), map_location="cpu", weights_only=True)
            except TypeError:
                if state is None:
                    state = torch.load(str(path), map_location="cpu")
        print(f"Loaded pretrained weight file: {path}")
    if not isinstance(state, dict) and hasattr(state, "state_dict"):
        state = state.state_dict()
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Weight source {source!r} did not contain a state dict.")
    return state


def _strip_prefix_if_present(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    matched = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    return matched if matched else state


def _filter_compatible_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    model_state = model.state_dict()
    compatible = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            compatible[key] = value
    return compatible


def _drop_common_checkpoint_prefix(key: str) -> str:
    """Remove wrappers commonly introduced by DDP or ReID model containers."""
    for prefix in ("module.", "model.", "base.", "backbone."):
        if key.startswith(prefix):
            return _drop_common_checkpoint_prefix(key[len(prefix) :])
    return key


def _normalize_checkpoint_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize common checkpoint wrappers without changing tensor content."""
    return {_drop_common_checkpoint_prefix(k): v for k, v in state.items()}


def _candidate_state_dicts(
    model: nn.Module,
    state: Dict[str, torch.Tensor],
    source: str,
) -> list[Tuple[str, Dict[str, torch.Tensor]]]:
    """Build state-dict candidates without cross-framework key remapping."""
    candidates: list[Tuple[str, Dict[str, torch.Tensor]]] = [("direct", state)]
    normalized = _normalize_checkpoint_state_dict(state)
    if set(normalized.keys()) != set(state.keys()):
        candidates.append(("common-prefix normalized", normalized))
    return candidates


def _overlap_stride_from_kernel(kernel_size: Tuple[int, int], ratio: float = 0.75) -> Tuple[int, int]:
    """Use overlapping ViT patches by keeping the kernel and reducing only stride."""
    return tuple(max(1, math.ceil(int(k) * float(ratio))) for k in kernel_size)


def _minimal_edge_cover_padding(
    input_size: Tuple[int, int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
) -> Tuple[int, int]:
    """Find the smallest symmetric padding that lets the last patch cover image edges."""
    padding = []
    for dim, kernel, step in zip(input_size, kernel_size, stride):
        dim = int(dim)
        kernel = int(kernel)
        step = int(step)
        for pad in range(max(kernel, step, dim) + 1):
            out = math.floor((dim + 2 * pad - kernel) / step) + 1
            if out <= 0:
                continue
            last_end = (out - 1) * step + kernel - pad
            if last_end >= dim:
                padding.append(pad)
                break
        else:
            raise ValueError(
                f"Cannot derive edge-cover padding for input={input_size}, "
                f"kernel={kernel_size}, stride={stride}."
            )
    return tuple(padding)


def _resolve_vit_reid_stem_geometry(
    conv: nn.Conv2d,
    stem_cfg: Dict[str, Any],
    input_size: Tuple[int, int],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Resolve ReID ViT stem stride and padding from explicit or automatic settings."""
    mode = str(stem_cfg.get("mode", "auto")).lower()
    if mode not in {"auto", "manual"}:
        raise ValueError(f"vit_stem.mode must be 'auto' or 'manual', got {mode!r}.")
    if mode == "manual" and "stride" not in stem_cfg:
        raise ValueError("vit_stem.mode='manual' requires vit_stem.stride.")

    kernel_size = tuple(stem_cfg["kernel_size"]) if "kernel_size" in stem_cfg else conv.kernel_size
    if tuple(kernel_size) != tuple(conv.kernel_size):
        raise ValueError(
            "ViT ReID stem keeps the pretrained patch kernel unchanged; "
            f"got kernel_size={kernel_size}, expected {conv.kernel_size}."
        )

    if "stride" in stem_cfg:
        stride = tuple(stem_cfg["stride"])
    else:
        ratio = float(stem_cfg.get("stride_ratio", 0.75))
        stride = _overlap_stride_from_kernel(conv.kernel_size, ratio)
    if any(s >= k for s, k in zip(stride, conv.kernel_size)):
        raise ValueError(f"ViT ReID stem requires stride < kernel_size, got stride={stride}, kernel={conv.kernel_size}.")

    padding = tuple(stem_cfg["padding"]) if "padding" in stem_cfg else _minimal_edge_cover_padding(
        input_size, conv.kernel_size, stride
    )
    return tuple(stride), tuple(padding)


class SideInformationEmbedding(nn.Module):
    """SIE token bias from one auxiliary index and optional automatic modality ids."""

    def __init__(self, dim: int, cfg: Dict[str, Any] | None, num_modalities: int, aux_cardinalities: Dict[int, int]):
        super().__init__()
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.scale = float(cfg.get("scale", 1.0))
        if "modality" in cfg:
            raise ValueError("SIE modality encoding is automatic; remove the modality field.")
        removed = sorted(set(cfg).intersection({"combine", "aux"}))
        if removed:
            raise ValueError(
                "Removed SIE field(s): "
                + ", ".join(removed)
                + ". Use model.encoder.sie.dim_index."
            )
        self.dim_index = int(cfg.get("dim_index", 0))
        self.aux_dims = [self.dim_index] if self.enabled else []
        self.use_modality = bool(cfg.get("_modality_sie", False))
        self.num_modalities = max(1, int(num_modalities))
        self.modality_embed = None
        self.embed = None
        if self.enabled:
            cardinality = int(cfg.get("num_values", aux_cardinalities.get(self.dim_index, 0)))
            if cardinality <= 0:
                raise ValueError(f"SIE dim_index {self.dim_index} has no inferred cardinality.")
            self.num_values = cardinality
            self.embed = nn.Embedding(cardinality, dim)
            if self.use_modality:
                self.modality_embed = nn.Embedding(self.num_modalities, dim)
        self._init_parameters(str(cfg.get("init", "zero")), float(cfg.get("std", 0.02)))

    def _init_parameters(self, init: str, std: float) -> None:
        if not self.enabled or self.embed is None:
            return
        embeddings = [self.embed]
        if self.modality_embed is not None:
            embeddings.append(self.modality_embed)
        for embedding in embeddings:
            if init == "zero":
                nn.init.zeros_(embedding.weight)
            elif init in {"trunc_normal", "normal"}:
                nn.init.trunc_normal_(embedding.weight, std=std)
            else:
                raise ValueError(f"Unknown SIE init: {init!r}")

    def forward(
        self,
        batch_size: int,
        device: torch.device,
        modality_ids: torch.Tensor | None = None,
        aux_indices: Dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        if not self.enabled:
            return None
        aux_indices = aux_indices or {}
        if self.dim_index not in aux_indices:
            raise ValueError(f"SIE dim_index {self.dim_index} is enabled but missing from batch aux_indices.")
        ids = aux_indices[self.dim_index].to(device=device, dtype=torch.long)
        if torch.any((ids < 0) | (ids >= int(self.num_values))):
            raise ValueError(f"SIE dim_index {self.dim_index} received ids outside [0, {int(self.num_values) - 1}].")
        out = self.embed(ids)
        if self.modality_embed is not None:
            if modality_ids is None:
                raise ValueError("Shared backbone SIE requires modality_ids.")
            modal_ids = modality_ids.to(device=device, dtype=torch.long)
            if torch.any((modal_ids < 0) | (modal_ids >= self.num_modalities)):
                raise ValueError(
                    f"SIE modality ids must be in [0, {self.num_modalities - 1}]."
                )
            out = out + self.modality_embed(modal_ids)
        return (self.scale * out).view(batch_size, 1, -1)


def _apply_optional_visual_projection(tokens: torch.Tensor, projection: Any) -> torch.Tensor:
    """Apply a real visual projection layer when the backbone exposes one."""
    if projection is None or isinstance(projection, nn.Identity):
        return tokens
    if isinstance(projection, nn.Linear):
        return projection(tokens)
    if torch.is_tensor(projection):
        proj = projection.to(device=tokens.device, dtype=tokens.dtype)
        return tokens @ proj
    if callable(projection):
        return projection(tokens)
    raise TypeError(f"Unsupported visual projection type: {type(projection).__name__}.")


class SideInformationEmbeddingRouter(nn.Module):
    """Dispatch one shared aux SIE table."""

    def __init__(
        self,
        dim: int,
        cfg: Dict[str, Any] | None,
        num_modalities: int,
        aux_cardinalities: Dict[int, int],
        shared: bool = True,
    ):
        super().__init__()
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.num_modalities = max(1, int(num_modalities))
        if not bool(shared):
            raise ValueError("Private SIE tables are not supported.")
        self.shared = True
        self.aux_dims = [int(cfg.get("dim_index", 0))] if self.enabled else []
        if not self.enabled:
            self.sie = None
            self.sie_modules = nn.ModuleList()
        else:
            self.sie = SideInformationEmbedding(dim, cfg, self.num_modalities, aux_cardinalities)
            self.sie_modules = nn.ModuleList()
            self.aux_dims = list(getattr(self.sie, "aux_dims", self.aux_dims))

    def forward(
        self,
        batch_size: int,
        device: torch.device,
        modality_ids: torch.Tensor | None = None,
        aux_indices: Dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        if not self.enabled:
            return None
        aux_indices = aux_indices or {}
        return self.sie(batch_size, device, modality_ids=modality_ids, aux_indices=aux_indices)


class ModalitySpecificBackboneAdapter(nn.Module):
    """Route each modality to a private encoder while keeping the public encoder API."""

    def __init__(self, encoders: list[nn.Module], local_modality_ids: bool = False):
        super().__init__()
        if not encoders:
            raise ValueError("ModalitySpecificBackboneAdapter requires at least one encoder.")
        self.encoders = nn.ModuleList(encoders)
        self.local_modality_ids = bool(local_modality_ids)
        dims = [int(getattr(encoder, "output_dim", 0)) for encoder in encoders]
        if not dims[0] or any(dim != dims[0] for dim in dims):
            raise ValueError(f"Private modality encoders must expose the same output_dim, got {dims}.")
        self.output_dim = dims[0]

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor | None = None,
        aux_indices: Dict[int, torch.Tensor] | None = None,
    ) -> BackboneOutput:
        if modality_ids is None:
            if len(self.encoders) == 1:
                return self.encoders[0](x, modality_ids=modality_ids, aux_indices=aux_indices)
            raise ValueError("Private modality encoders require modality_ids.")

        ids = modality_ids.to(device=x.device, dtype=torch.long).flatten()
        if ids.numel() != x.shape[0]:
            raise ValueError(f"modality_ids length {ids.numel()} does not match batch size {x.shape[0]}.")
        if torch.any((ids < 0) | (ids >= len(self.encoders))):
            raise ValueError(f"modality_ids must be in [0, {len(self.encoders) - 1}].")

        cls_out = None
        spatial_out = None
        spatial_hw = None
        aux_indices = aux_indices or {}
        for idx in torch.unique(ids).tolist():
            idx = int(idx)
            mask = ids == idx
            sub_aux = {dim: value.to(device=x.device)[mask] for dim, value in aux_indices.items()}
            routed_ids = torch.zeros_like(ids[mask]) if self.local_modality_ids else ids[mask]
            out = self.encoders[idx](x[mask], modality_ids=routed_ids, aux_indices=sub_aux)
            if cls_out is None:
                cls_out = out.cls_pre.new_empty((x.shape[0], out.cls_pre.shape[-1]))
                if out.spatial_tokens is not None:
                    spatial_out = out.spatial_tokens.new_empty(
                        (x.shape[0], out.spatial_tokens.shape[1], out.spatial_tokens.shape[2])
                    )
                    spatial_hw = out.spatial_hw
            cls_out[mask] = out.cls_pre
            if out.spatial_tokens is not None:
                if spatial_out is None:
                    raise ValueError("Private encoders returned inconsistent spatial token availability.")
                if spatial_hw != out.spatial_hw:
                    raise ValueError("Private encoders returned inconsistent spatial token shapes.")
                spatial_out[mask] = out.spatial_tokens
        if cls_out is None:
            raise RuntimeError("Empty batch cannot be routed through private modality encoders.")
        return BackboneOutput(cls_pre=cls_out, spatial_tokens=spatial_out, spatial_hw=spatial_hw)


def load_state_dict_from_path_or_url(model: nn.Module, source: str, prefix: str | None = None) -> None:
    """Load compatible weights from a local path, safetensors file, or URL."""
    state = _load_state_dict_source(source)
    total_tensors = sum(1 for value in state.values() if torch.is_tensor(value))
    if prefix:
        state = _strip_prefix_if_present(state, prefix)
    compatible = {}
    strategy = "direct"
    for candidate_strategy, candidate_state in _candidate_state_dicts(model, state, str(source)):
        candidate_compatible = _filter_compatible_state_dict(model, candidate_state)
        if candidate_compatible:
            compatible = candidate_compatible
            strategy = candidate_strategy
            break
    if not compatible:
        raise RuntimeError(
            "No compatible parameters found in pretrained weight source. "
            f"Source: {source}. Check backbone name/source, CLIP visual prefix, and checkpoint format."
        )
    model.load_state_dict(compatible, strict=False)
    skipped = max(0, total_tensors - len(compatible))
    if strategy != "direct":
        print(f"Applying pretrained checkpoint key adaptation: {strategy}.")
    print(
        "Loaded pretrained weights into backbone: "
        f"{len(compatible)}/{total_tensors} tensors matched; "
        f"matched={len(compatible)}, skipped={skipped}, total={total_tensors}."
    )
    if skipped:
        print(
            "Warning: partial compatible pretrained load; skipped tensors were ignored "
            "because their names or shapes do not match the current backbone."
        )


class ImageBackboneAdapter(nn.Module):
    """Wrap an image model and expose MV-SRBF backbone outputs."""

    def __init__(
        self,
        model: nn.Module,
        input_size: Tuple[int, int] = (128, 256),
        feature_dim: int | None = None,
        last_stride: int | None = None,
        vit_stem: Dict[str, Any] | None = None,
        sie_cfg: Dict[str, Any] | None = None,
        num_modalities: int = 1,
        sie_shared: bool = True,
    ):
        super().__init__()
        self.num_modalities = max(1, int(num_modalities))
        if last_stride is not None:
            self._apply_last_stride(model, int(last_stride))
        self.is_vit = self._is_torchvision_vit(model)
        if self.is_vit:
            self._apply_vit_reid_stem(model, vit_stem, input_size)
        self.sie = (
            SideInformationEmbeddingRouter(
                model.conv_proj.out_channels,
                sie_cfg,
                int((sie_cfg or {}).get("num_modalities", 1)),
                dict((sie_cfg or {}).get("aux_cardinalities", {})),
                shared=sie_shared,
            )
            if self.is_vit
            else None
        )
        self.trunk = model if self.is_vit else self._remove_classifier(model)
        inferred_dim = self._infer_feature_dim(input_size)
        if feature_dim is not None and int(feature_dim) != inferred_dim:
            raise ValueError(
                f"model.encoder.backbone.feature_dim={int(feature_dim)} does not match raw backbone output "
                f"dimension {inferred_dim}. MV-SRBF no longer adds an extra embedding projection; "
                "remove feature_dim or set it to the true backbone output width."
            )
        self.output_dim = inferred_dim
        self.pool = nn.AdaptiveAvgPool2d(1)

    def _is_torchvision_vit(self, model: nn.Module) -> bool:
        return all(hasattr(model, attr) for attr in ("conv_proj", "class_token", "encoder"))

    def _apply_vit_reid_stem(
        self,
        model: nn.Module,
        stem_cfg: Dict[str, Any] | None,
        input_size: Tuple[int, int],
    ) -> None:
        """Keep ViT patch weights and reduce stride for overlapping ReID patches."""
        cfg = stem_cfg or {}
        if not bool(cfg.get("enabled", False)):
            return
        if not isinstance(model.conv_proj, nn.Conv2d):
            raise TypeError("ViT ReID stem expects torchvision ViT conv_proj to be nn.Conv2d.")

        conv = model.conv_proj
        out_channels = int(cfg.get("out_channels", conv.out_channels))
        if out_channels != conv.out_channels:
            raise ValueError(
                f"ViT stem out_channels={out_channels} must match transformer hidden dim {conv.out_channels}."
            )
        stride, padding = _resolve_vit_reid_stem_geometry(conv, cfg, input_size)
        conv.stride = tuple(stride)
        conv.padding = tuple(padding)

    def _derive_vit_reid_stride(
        self,
        kernel_size: Tuple[int, int],
        cfg: Dict[str, Any],
    ) -> Tuple[int, int]:
        ratio = float(cfg.get("stride_ratio", 0.75))
        stride = _overlap_stride_from_kernel(kernel_size, ratio)
        if any(s >= k for s, k in zip(stride, kernel_size)):
            raise ValueError(f"ViT ReID stem requires stride < kernel_size, got stride={stride}, kernel={kernel_size}.")
        return stride

    def _apply_last_stride(self, model: nn.Module, last_stride: int) -> None:
        """Apply a unified ReID last-stride setting to CNN feature extractors."""
        if last_stride <= 0:
            raise ValueError(f"last_stride must be a positive integer or null, got {last_stride}.")
        if self._looks_like_token_backbone(model):
            return
        if self._apply_resnet_last_stride(model, last_stride):
            return
        self._apply_feature_container_last_stride(model, last_stride)

    def _looks_like_token_backbone(self, model: nn.Module) -> bool:
        """Avoid treating transformer patch embedding stride as CNN last stride."""
        class_name = model.__class__.__name__.lower()
        if self._is_torchvision_vit(model):
            return True
        if hasattr(model, "patch_embed") and hasattr(model, "blocks"):
            return True
        return any(token in class_name for token in ("visiontransformer", "swin", "maxvit"))

    def _apply_resnet_last_stride(self, model: nn.Module, last_stride: int) -> bool:
        if not hasattr(model, "layer4"):
            return False
        try:
            block0 = model.layer4[0]
            target_stride = (last_stride, last_stride)
            if hasattr(block0, "conv2") and block0.conv2.stride != (1, 1):
                block0.conv2.stride = target_stride
            elif hasattr(block0, "conv1") and block0.conv1.stride != (1, 1):
                block0.conv1.stride = target_stride
            elif hasattr(block0, "conv2"):
                block0.conv2.stride = (last_stride, last_stride)
            if getattr(block0, "downsample", None) is not None:
                block0.downsample[0].stride = target_stride
            return True
        except (TypeError, IndexError, AttributeError):
            return False

    def _apply_feature_container_last_stride(self, model: nn.Module, last_stride: int) -> bool:
        """Set the final editable downsampling stride in generic CNN features."""
        if not hasattr(model, "features") or not isinstance(model.features, nn.Module):
            return False
        candidates = []
        for name, module in model.features.named_modules():
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                stride = self._module_stride_tuple(module)
                if any(value > 1 for value in stride):
                    candidates.append((name, module, stride))
        if not candidates:
            return False
        name, module, old_stride = candidates[-1]
        target_stride = (last_stride, last_stride)
        module.stride = target_stride
        print(
            f"Applied last_stride={last_stride} to {model.__class__.__name__}.features.{name}: "
            f"{old_stride} -> {target_stride}."
        )
        return True

    @staticmethod
    def _module_stride_tuple(module: nn.Module) -> Tuple[int, int]:
        stride = getattr(module, "stride", (1, 1))
        if stride is None:
            stride = getattr(module, "kernel_size", (1, 1))
        if isinstance(stride, tuple):
            return int(stride[0]), int(stride[1])
        return int(stride), int(stride)

    def _forward_vit_tokens(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor | None = None,
        aux_indices: Dict[int, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        seq = self.trunk.conv_proj(x)
        batch_size, hidden_dim, h, w = seq.shape
        seq = seq.reshape(batch_size, hidden_dim, h * w).permute(0, 2, 1)
        cls_token = self.trunk.class_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls_token, seq], dim=1)
        seq = seq + self._vit_pos_embedding(h, w).to(dtype=seq.dtype, device=seq.device)
        sie = self.sie(batch_size, seq.device, modality_ids=modality_ids, aux_indices=aux_indices)
        if sie is not None:
            seq = seq + sie.to(dtype=seq.dtype)
        seq = self.trunk.encoder.dropout(seq)
        seq = self.trunk.encoder.layers(seq)
        seq = self.trunk.encoder.ln(seq)
        seq = _apply_optional_visual_projection(seq, getattr(self.trunk, "proj", None))
        return seq[:, 0], seq[:, 1:], (h, w)

    def _vit_pos_embedding(self, h: int, w: int) -> torch.Tensor:
        pos = self.trunk.encoder.pos_embedding
        if pos.shape[1] == h * w + 1:
            return pos
        cls_pos = pos[:, :1]
        patch_pos = pos[:, 1:]
        old_hw = int(patch_pos.shape[1] ** 0.5)
        patch_pos = patch_pos.reshape(1, old_hw, old_hw, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(h, w), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def _remove_classifier(self, model: nn.Module) -> nn.Module:
        if hasattr(model, "features") and isinstance(model.features, nn.Module):
            return model.features
        if hasattr(model, "conv1") and hasattr(model, "layer4"):
            children = []
            for name, child in model.named_children():
                if name in {"avgpool", "fc", "classifier", "head", "heads"}:
                    break
                children.append(child)
            if children:
                return nn.Sequential(*children)
        for attr in ("fc", "classifier", "head", "heads"):
            if hasattr(model, attr) and isinstance(getattr(model, attr), nn.Module):
                setattr(model, attr, nn.Identity())
        return model

    @torch.no_grad()
    def _infer_feature_dim(self, input_size: Tuple[int, int]) -> int:
        was_training = self.trunk.training
        self.trunk.eval()
        dummy = torch.zeros(1, 3, int(input_size[0]), int(input_size[1]))
        if self.is_vit:
            aux = {dim_id: torch.zeros(1, dtype=torch.long) for dim_id in getattr(self.sie, "aux_dims", [])}
            out, _, _ = self._forward_vit_tokens(dummy, modality_ids=torch.zeros(1, dtype=torch.long), aux_indices=aux)
        else:
            out = self.trunk(dummy)
        if isinstance(out, (tuple, list)):
            out = out[0]
        dim = int(out.shape[1]) if out.ndim == 4 else int(out.flatten(1).shape[1])
        self.trunk.train(was_training)
        return dim

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor | None = None,
        aux_indices: Dict[int, torch.Tensor] | None = None,
    ) -> BackboneOutput:
        if self.is_vit:
            cls, patches, spatial_hw = self._forward_vit_tokens(x, modality_ids=modality_ids, aux_indices=aux_indices)
            return BackboneOutput(
                cls_pre=cls,
                spatial_tokens=patches,
                spatial_hw=spatial_hw,
            )
        feats = self.trunk(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if feats.ndim == 4:
            spatial_hw = (int(feats.shape[2]), int(feats.shape[3]))
            spatial = feats.flatten(2).transpose(1, 2)
            pooled = self.pool(feats).flatten(1)
            return BackboneOutput(
                cls_pre=pooled,
                spatial_tokens=spatial,
                spatial_hw=spatial_hw,
            )
        return BackboneOutput(cls_pre=feats.flatten(1), spatial_tokens=None)


class CLIPVisionBackboneAdapter(nn.Module):
    """Wrap a CLIP ViT visual encoder and expose MV-SRBF backbone outputs."""

    def __init__(
        self,
        visual: nn.Module,
        input_size: Tuple[int, int] = (128, 256),
        feature_dim: int | None = None,
        vit_stem: Dict[str, Any] | None = None,
        sie_cfg: Dict[str, Any] | None = None,
        num_modalities: int = 1,
        sie_shared: bool = True,
    ):
        super().__init__()
        if not self._is_clip_vit(visual):
            raise TypeError("CLIP backbone currently supports ViT visual encoders only.")
        self.num_modalities = max(1, int(num_modalities))
        self.trunk = visual
        self._apply_clip_vit_reid_stem(vit_stem, input_size)
        self.sie = SideInformationEmbeddingRouter(
            self.trunk.conv1.out_channels,
            sie_cfg,
            int((sie_cfg or {}).get("num_modalities", 1)),
            dict((sie_cfg or {}).get("aux_cardinalities", {})),
            shared=sie_shared,
        )
        inferred_dim = self._infer_feature_dim(input_size)
        if feature_dim is not None and int(feature_dim) != inferred_dim:
            raise ValueError(
                f"model.encoder.backbone.feature_dim={int(feature_dim)} does not match raw CLIP visual output "
                f"dimension {inferred_dim}. MV-SRBF keeps only the native CLIP visual projection; "
                "remove feature_dim or set it to the true visual output width."
            )
        self.output_dim = inferred_dim

    def _is_clip_vit(self, visual: nn.Module) -> bool:
        return all(
            hasattr(visual, attr)
            for attr in ("conv1", "class_embedding", "positional_embedding", "transformer")
        )

    def _apply_clip_vit_reid_stem(self, stem_cfg: Dict[str, Any] | None, input_size: Tuple[int, int]) -> None:
        cfg = stem_cfg or {}
        if not bool(cfg.get("enabled", False)):
            return
        conv = self.trunk.conv1
        if not isinstance(conv, nn.Conv2d):
            raise TypeError("CLIP ViT ReID stem expects visual.conv1 to be nn.Conv2d.")
        stride, padding = _resolve_vit_reid_stem_geometry(conv, cfg, input_size)
        conv.stride = tuple(stride)
        conv.padding = tuple(padding)

    def _derive_vit_reid_stride(
        self,
        kernel_size: Tuple[int, int],
        cfg: Dict[str, Any],
    ) -> Tuple[int, int]:
        return ImageBackboneAdapter._derive_vit_reid_stride(self, kernel_size, cfg)

    def _clip_pos_embedding(self, h: int, w: int) -> torch.Tensor:
        pos = self.trunk.positional_embedding
        if pos.ndim == 2:
            pos = pos.unsqueeze(0)
        if pos.shape[1] == h * w + 1:
            return pos
        cls_pos = pos[:, :1]
        patch_pos = pos[:, 1:]
        old_hw = int(patch_pos.shape[1] ** 0.5)
        patch_pos = patch_pos.reshape(1, old_hw, old_hw, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(h, w), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        return torch.cat([cls_pos, patch_pos], dim=1).squeeze(0)

    def _forward_clip_vit_tokens(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor | None = None,
        aux_indices: Dict[int, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        x = self.trunk.conv1(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        cls = self.trunk.class_embedding.to(dtype=x.dtype, device=x.device)
        cls = cls.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = self._clip_pos_embedding(h, w).to(dtype=x.dtype, device=x.device)
        x = x + pos
        sie = self.sie(b, x.device, modality_ids=modality_ids, aux_indices=aux_indices)
        if sie is not None:
            x = x + sie.to(dtype=x.dtype)
        if hasattr(self.trunk, "ln_pre"):
            x = self.trunk.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.trunk.transformer(x)
        x = x.permute(1, 0, 2)
        if hasattr(self.trunk, "ln_post"):
            x = self.trunk.ln_post(x)
        x = _apply_optional_visual_projection(x, getattr(self.trunk, "proj", None))
        return x[:, 0], x[:, 1:], (h, w)

    @torch.no_grad()
    def _infer_feature_dim(self, input_size: Tuple[int, int]) -> int:
        was_training = self.trunk.training
        self.trunk.eval()
        dummy = torch.zeros(1, 3, int(input_size[0]), int(input_size[1]))
        aux = {dim_id: torch.zeros(1, dtype=torch.long) for dim_id in getattr(self.sie, "aux_dims", [])}
        cls, _, _ = self._forward_clip_vit_tokens(dummy, modality_ids=torch.zeros(1, dtype=torch.long), aux_indices=aux)
        self.trunk.train(was_training)
        return int(cls.shape[1])

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor | None = None,
        aux_indices: Dict[int, torch.Tensor] | None = None,
    ) -> BackboneOutput:
        cls, patches, spatial_hw = self._forward_clip_vit_tokens(x, modality_ids=modality_ids, aux_indices=aux_indices)
        return BackboneOutput(
            cls_pre=cls,
            spatial_tokens=patches,
            spatial_hw=spatial_hw,
        )


class DinoVisionBackboneAdapter(nn.Module):
    """Adapt official DINO ViTs, including DINOv3 register-token Memory readout."""

    def __init__(
        self,
        model: nn.Module,
        *,
        version: str,
        input_size: Tuple[int, int] = (128, 256),
        feature_dim: int | None = None,
        memory_cfg: Dict[str, Any] | None = None,
        sie_cfg: Dict[str, Any] | None = None,
        num_modalities: int = 1,
        sie_shared: bool = True,
    ):
        super().__init__()
        self.trunk = model
        self.version = str(version).lower()
        if self.version not in {"dinov2", "dinov3"}:
            raise ValueError(f"Unsupported DINO version: {version!r}.")
        memory_cfg = dict(memory_cfg or {})
        self.memory_enabled = bool(memory_cfg.get("enabled", False))
        if self.memory_enabled and self.version != "dinov3":
            raise ValueError("DINO register-token Memory mode is supported only for DINOv3.")
        self.memory_aggregation = str(memory_cfg.get("aggregation", "mean")).strip().lower()
        if self.memory_aggregation != "mean":
            raise ValueError("DINOv3 memory.aggregation must be 'mean'.")
        inferred_dim = int(getattr(model, "embed_dim", 0) or getattr(model, "num_features", 0) or 0)
        if inferred_dim <= 0:
            inferred_dim = self._infer_feature_dim(input_size)
        if feature_dim is not None and int(feature_dim) != inferred_dim:
            raise ValueError(
                f"model.encoder.backbone.feature_dim={int(feature_dim)} does not match "
                f"the DINO backbone output width {inferred_dim}."
            )
        self.output_dim = inferred_dim
        self.sie = SideInformationEmbeddingRouter(
            inferred_dim,
            sie_cfg,
            int((sie_cfg or {}).get("num_modalities", num_modalities)),
            dict((sie_cfg or {}).get("aux_cardinalities", {})),
            shared=sie_shared,
        )

    def _run_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        forward_features = getattr(self.trunk, "forward_features", None)
        if not callable(forward_features):
            raise TypeError("DINO backbone must expose forward_features(x).")
        out = forward_features(x)
        if not isinstance(out, dict):
            raise TypeError("DINO forward_features must return a feature dictionary.")
        return out

    def _patch_size(self) -> Tuple[int, int] | None:
        patch_size = getattr(self.trunk, "patch_size", None)
        if patch_size is None and hasattr(self.trunk, "patch_embed"):
            patch_size = getattr(self.trunk.patch_embed, "patch_size", None)
        if isinstance(patch_size, int):
            return patch_size, patch_size
        if isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
            return int(patch_size[0]), int(patch_size[1])
        return None

    def _pad_to_patch_grid(self, x: torch.Tensor) -> torch.Tensor:
        patch_size = self._patch_size()
        if patch_size is None:
            return x
        patch_h, patch_w = patch_size
        pad_h = (-int(x.shape[-2])) % patch_h
        pad_w = (-int(x.shape[-1])) % patch_w
        if not pad_h and not pad_w:
            return x
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

    def _extract_tokens(
        self,
        out: Dict[str, torch.Tensor],
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, Tuple[int, int] | None]:
        cls = out.get("x_norm_clstoken")
        patches = out.get("x_norm_patchtokens")
        if not torch.is_tensor(cls):
            raise KeyError("DINO features are missing x_norm_clstoken.")
        if patches is not None and not torch.is_tensor(patches):
            raise TypeError("DINO x_norm_patchtokens must be a tensor.")
        if self.memory_enabled:
            register_tokens = out.get("x_storage_tokens")
            if not torch.is_tensor(register_tokens) or register_tokens.ndim != 3 or register_tokens.shape[1] == 0:
                raise ValueError(
                    "DINOv3 register-token Memory mode requires non-empty x_storage_tokens."
                )
            cls = torch.cat((cls.unsqueeze(1), register_tokens), dim=1).mean(dim=1)
        spatial_hw = self._spatial_hw(x, patches)
        return cls, patches, spatial_hw

    def _spatial_hw(
        self,
        x: torch.Tensor,
        patches: torch.Tensor | None,
    ) -> Tuple[int, int] | None:
        if patches is None or patches.ndim != 3:
            return None
        patch_size = self._patch_size()
        if patch_size is not None:
            h = int(x.shape[-2]) // int(patch_size[0])
            w = int(x.shape[-1]) // int(patch_size[1])
            if h * w == int(patches.shape[1]):
                return h, w
        side = int(math.isqrt(int(patches.shape[1])))
        return (side, side) if side * side == int(patches.shape[1]) else None

    @torch.no_grad()
    def _infer_feature_dim(self, input_size: Tuple[int, int]) -> int:
        was_training = self.trunk.training
        self.trunk.eval()
        dummy = self._pad_to_patch_grid(torch.zeros(1, 3, int(input_size[0]), int(input_size[1])))
        cls, _, _ = self._extract_tokens(self._run_features(dummy), dummy)
        self.trunk.train(was_training)
        return int(cls.shape[-1])

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor | None = None,
        aux_indices: Dict[int, torch.Tensor] | None = None,
    ) -> BackboneOutput:
        model_input = self._pad_to_patch_grid(x)
        cls, patches, spatial_hw = self._extract_tokens(self._run_features(model_input), model_input)
        sie = self.sie(x.shape[0], x.device, modality_ids=modality_ids, aux_indices=aux_indices)
        if sie is not None:
            cls = cls + sie[:, 0].to(dtype=cls.dtype)
            if patches is not None:
                patches = patches + sie.to(dtype=patches.dtype)
        return BackboneOutput(cls_pre=cls, spatial_tokens=patches, spatial_hw=spatial_hw)


class HuggingFaceVisionBackboneAdapter(nn.Module):
    """Wrap HuggingFace/Transformers vision models that expose hidden states."""

    def __init__(self, model: nn.Module, input_size: Tuple[int, int] = (128, 256)):
        super().__init__()
        self.trunk = model
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = self._infer_feature_dim(input_size)

    def _run_model(self, x: torch.Tensor):
        try:
            return self.trunk(pixel_values=x)
        except TypeError:
            return self.trunk(x)

    def _to_backbone_output(self, out: Any) -> BackboneOutput:
        hidden = getattr(out, "last_hidden_state", None)
        pooler = getattr(out, "pooler_output", None)
        if hidden is None:
            if isinstance(out, dict):
                hidden = out.get("last_hidden_state") or out.get("features")
                pooler = out.get("pooler_output", pooler)
            elif isinstance(out, (tuple, list)) and out:
                hidden = out[0]
        if hidden is None:
            hidden = out
        if not torch.is_tensor(hidden):
            raise TypeError("HuggingFace vision backbone did not return tensor hidden states.")
        if hidden.ndim == 3:
            cls = pooler if torch.is_tensor(pooler) else hidden[:, 0]
            spatial = hidden[:, 1:] if hidden.shape[1] > 1 else None
            return BackboneOutput(cls_pre=cls, spatial_tokens=spatial)
        if hidden.ndim == 4:
            spatial_hw = (int(hidden.shape[2]), int(hidden.shape[3]))
            spatial = hidden.flatten(2).transpose(1, 2)
            pooled = self.pool(hidden).flatten(1)
            return BackboneOutput(cls_pre=pooled, spatial_tokens=spatial, spatial_hw=spatial_hw)
        return BackboneOutput(cls_pre=hidden.flatten(1), spatial_tokens=None)

    @torch.no_grad()
    def _infer_feature_dim(self, input_size: Tuple[int, int]) -> int:
        was_training = self.trunk.training
        self.trunk.eval()
        dummy = torch.zeros(1, 3, int(input_size[0]), int(input_size[1]))
        out = self._to_backbone_output(self._run_model(dummy))
        self.trunk.train(was_training)
        return int(out.cls_pre.shape[1])

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor | None = None,
        aux_indices: Dict[int, torch.Tensor] | None = None,
    ) -> BackboneOutput:
        return self._to_backbone_output(self._run_model(x))


def build_torchvision_model(name: str, pretrained: bool) -> nn.Module:
    """Build a model from torchvision.models by exact or normalized name."""
    import torchvision.models as tv_models

    resolved = normalize_model_name(name)
    factory = getattr(tv_models, resolved, None)
    if factory is None:
        available = []
        if hasattr(tv_models, "list_models"):
            available = sorted(tv_models.list_models())
        hint = ", ".join(available[:20])
        raise ValueError(f"torchvision.models has no model named {name!r} (resolved {resolved!r}). Examples: {hint}")

    kwargs: Dict[str, Any] = {}
    signature = inspect.signature(factory)
    if "weights" in signature.parameters:
        if pretrained:
            try:
                from torchvision.models import get_model_weights

                kwargs["weights"] = get_model_weights(resolved).DEFAULT
            except Exception:
                kwargs["weights"] = "DEFAULT"
        else:
            kwargs["weights"] = None
    elif "pretrained" in signature.parameters:
        kwargs["pretrained"] = pretrained
    if resolved == "inception_v3":
        kwargs.setdefault("aux_logits", False)
    return factory(**kwargs)


def build_timm_model(name: str, pretrained: bool, kwargs: Dict[str, Any] | None = None) -> nn.Module:
    """Build a model directly through timm.create_model."""
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm backbone source requires the optional dependency 'timm'. "
            "Install it with `pip install timm`, or use source='torchvision'/'custom'."
        ) from exc

    kwargs = dict(kwargs or {})
    kwargs.setdefault("num_classes", 0)
    model = timm.create_model(name, pretrained=pretrained, **kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError(f"timm model {name!r} did not return nn.Module.")
    return model


def build_torch_hub_model(repo: str, name: str, pretrained: bool, kwargs: Dict[str, Any] | None = None) -> nn.Module:
    """Build a model through torch.hub.load."""
    model = torch.hub.load(repo, name, pretrained=pretrained, **(kwargs or {}))
    if not isinstance(model, nn.Module):
        raise TypeError(f"torch.hub model {repo}:{name} did not return nn.Module.")
    return model


def build_huggingface_model(name: str, pretrained: bool | str = True, kwargs: Dict[str, Any] | None = None) -> nn.Module:
    """Build a HuggingFace vision model through transformers.AutoModel."""
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:
        raise ImportError(
            "HuggingFace/Transformers backbone source requires `transformers`. "
            "Install it with `pip install transformers`, or use source='custom'."
        ) from exc
    kwargs = dict(kwargs or {})
    if pretrained is False:
        config = AutoConfig.from_pretrained(name, **kwargs)
        model = AutoModel.from_config(config)
    else:
        model = AutoModel.from_pretrained(name, **kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError(f"transformers AutoModel {name!r} did not return nn.Module.")
    return model


def build_custom_model(target: str, cfg: Dict[str, Any]) -> nn.Module:
    """Build a custom model from an import target using optional kwargs."""
    factory = import_object(target)
    model = factory(**cfg.get("kwargs", {}))
    if not isinstance(model, nn.Module):
        raise TypeError(f"Custom backbone target {target!r} did not return nn.Module.")
    return model


# Known OpenAI CLIP ViT visual encoder architectures.
# These allow building the model structure without downloading pretrained weights.
_OPENAI_CLIP_VIT_CONFIGS: Dict[str, Dict[str, int]] = {
    "ViT-B/32": {"output_dim": 512, "input_resolution": 224, "width": 768, "layers": 12, "heads": 12, "patch_size": 32},
    "ViT-B/16": {"output_dim": 512, "input_resolution": 224, "width": 768, "layers": 12, "heads": 12, "patch_size": 16},
    "ViT-L/14": {"output_dim": 768, "input_resolution": 224, "width": 1024, "layers": 24, "heads": 16, "patch_size": 14},
    "ViT-L/14@336px": {"output_dim": 768, "input_resolution": 336, "width": 1024, "layers": 24, "heads": 16, "patch_size": 14},
}


def _build_openai_clip_visual_from_package(
    name: str,
    kwargs: Dict[str, Any],
    weights_source: str | None = None,
) -> nn.Module:
    """Use the OpenAI CLIP package as the source of truth for OpenAI CLIP weights."""
    clip_source = str(weights_source) if weights_source else str(name)
    try:
        import clip
    except ImportError as exc:
        raise ImportError(
            "source='openai-clip' requires the OpenAI CLIP package. Install it with "
            "`pip install git+https://github.com/openai/CLIP.git` or use source='open-clip'."
        ) from exc
    load_kwargs = {}
    if kwargs.get("download_root"):
        load_kwargs["download_root"] = kwargs["download_root"]
    model, _ = clip.load(clip_source, device="cpu", jit=False, **load_kwargs)
    visual = getattr(model, "visual", None)
    if not isinstance(visual, nn.Module):
        raise TypeError(f"openai-clip model {clip_source!r} did not expose an nn.Module visual encoder.")
    setattr(visual, "_mvsrbf_pretrained_source", str(weights_source or f"openai-clip:{clip_source}"))
    print(f"Loaded CLIP visual backbone through openai-clip: source={clip_source}")
    return visual


def _build_openai_clip_visual_random_init(
    name: str,
    kwargs: Dict[str, Any],
) -> nn.Module:
    """Build an OpenAI CLIP visual encoder with random init (no weight download).

    Uses the ``VisionTransformer`` class from ``clip.model`` directly, constructing
    the architecture from a known configuration without calling ``clip.load()``.
    """
    try:
        from clip.model import VisionTransformer
    except ImportError as exc:
        raise ImportError(
            "source='openai-clip' requires the OpenAI CLIP package. Install it with "
            "`pip install git+https://github.com/openai/CLIP.git` or use source='open-clip'."
        ) from exc
    cfg = _OPENAI_CLIP_VIT_CONFIGS.get(name)
    if cfg is None:
        known = ", ".join(sorted(_OPENAI_CLIP_VIT_CONFIGS))
        raise ValueError(
            f"Cannot build random-init OpenAI CLIP model {name!r}: "
            f"architecture not in known configs ({known}). "
            f"Use pretrained: true or source='open-clip' for this model."
        )
    visual = VisionTransformer(**cfg)
    setattr(visual, "_mvsrbf_pretrained_source", "random-init")
    print(f"Built random-init CLIP visual backbone: name={name}")
    return visual


def build_openai_clip_vision_model(
    name: str,
    pretrained: bool | str = True,
    kwargs: Dict[str, Any] | None = None,
    weights_source: str | None = None,
) -> nn.Module:
    """Build an OpenAI CLIP visual encoder through the official `clip` package.

    When ``pretrained`` is ``False`` and no ``weights_source`` is given, the
    visual encoder is built directly from ``clip.model.VisionTransformer``
    with random initialisation — no model weights are downloaded.
    """
    kwargs = kwargs or {}
    if pretrained is False and not weights_source:
        return _build_openai_clip_visual_random_init(name, kwargs)
    return _build_openai_clip_visual_from_package(name, kwargs, weights_source=weights_source)


def build_open_clip_vision_model(
    name: str,
    pretrained: bool | str = True,
    kwargs: Dict[str, Any] | None = None,
    weights_source: str | None = None,
) -> nn.Module:
    """Build an open_clip visual encoder through `open_clip_torch` only."""
    kwargs = kwargs or {}
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "source='open_clip' requires the optional dependency 'open_clip_torch'. "
            "Install it with `pip install open_clip_torch`, or use source='openai-clip' for OpenAI CLIP."
        ) from exc

    pretrained_tag = weights_source if weights_source else ("openai" if pretrained is True else pretrained)
    if pretrained is False:
        pretrained_tag = None
    if weights_source:
        pretrained_tag = weights_source
    if pretrained_tag:
        print(f"Loaded CLIP visual backbone through open_clip: name={name}, pretrained={pretrained_tag}")
    else:
        print(f"Initialized random-init CLIP visual backbone through open_clip: name={name}")
    model = open_clip.create_model(name, pretrained=pretrained_tag, **kwargs)
    visual = getattr(model, "visual", None)
    if not isinstance(visual, nn.Module):
        raise TypeError(f"open_clip model {name!r} did not expose an nn.Module visual encoder.")
    if weights_source:
        setattr(visual, "_mvsrbf_pretrained_source", str(weights_source))
    return visual
