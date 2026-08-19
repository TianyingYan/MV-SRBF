"""torchvision-first backbone factory for MV-SRBF."""
from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import torch.nn as nn

from utils.input_config import resolve_padded_input_size
from utils.config import resolve_runtime_path
from .backbone_adapter import (
    CLIPVisionBackboneAdapter,
    DinoVisionBackboneAdapter,
    HuggingFaceVisionBackboneAdapter,
    ImageBackboneAdapter,
    ModalitySpecificBackboneAdapter,
    build_custom_model,
    build_huggingface_model,
    build_open_clip_vision_model,
    build_openai_clip_vision_model,
    build_timm_model,
    build_torch_hub_model,
    build_torchvision_model,
    load_state_dict_from_path_or_url,
)
from utils.weight_init import apply_model_init


def infer_backbone_family(name: str, cfg_family: str | None = None) -> str:
    """Infer the neck family used by the ReID head."""
    if cfg_family:
        return cfg_family
    lname = name.lower()
    if "vit" in lname or "swin" in lname or "transformer" in lname:
        return "vit"
    return "cnn"


def resolve_vit_stem_config(cfg: Dict[str, Any]) -> Dict[str, Any] | None:
    """Return ViT ReID stem config; auto geometry is derived after model build."""
    mcfg = cfg["model"]["encoder"]["backbone"]
    stem_cfg = mcfg.get("vit_stem")
    if stem_cfg is None:
        return None
    return dict(stem_cfg)


def resolve_num_modalities(cfg: Dict[str, Any]) -> int:
    """Resolve the dataset modality count from inferred metadata first."""
    dataset_cfg = cfg.get("dataset", {})
    metadata = dataset_cfg.get("metadata", {})
    modal_keys = metadata.get("modal_keys") or list(dataset_cfg.get("modalities", {}).keys())
    return int(metadata.get("num_modalities") or dataset_cfg.get("num_modalities", len(modal_keys) or 1))


def resolve_sie_config(cfg: Dict[str, Any]) -> Dict[str, Any] | None:
    """Resolve the single-aux-dimension SIE config from dataset metadata."""
    mcfg = cfg["model"]["encoder"]
    raw = mcfg.get("sie")
    if raw is None or not bool(raw.get("enabled", False)):
        return None
    if "modality" in raw:
        raise ValueError(
            "model.encoder.sie.modality is automatic: shared backbones add modality SIE; "
            "private backbones do not. Remove the modality field."
        )
    removed = sorted(set(raw).intersection({"combine", "aux"}))
    if removed:
        raise ValueError(
            "Removed SIE field(s): "
            + ", ".join(removed)
            + ". Use model.encoder.sie.dim_index with one dataset aux dimension."
        )
    if raw.get("dim_index") is None:
        raise ValueError("model.encoder.sie.dim_index is required when SIE is enabled.")
    sie_cfg = dict(raw)
    sharing = mcfg.get("sharing", {})
    sie_cfg["_modality_sie"] = bool(sharing.get("backbone", True))
    dataset_cfg = cfg.get("dataset", {})
    metadata = dataset_cfg.get("metadata", {})
    modal_keys = metadata.get("modal_keys") or list(dataset_cfg.get("modalities", {}).keys())
    num_modalities = int(metadata.get("num_modalities") or dataset_cfg.get("num_modalities", len(modal_keys) or 1))
    sie_cfg["num_modalities"] = num_modalities

    dim_index = int(sie_cfg["dim_index"])
    aux_meta = metadata.get("aux_dims", {})
    meta = aux_meta.get(dim_index, aux_meta.get(str(dim_index), {}))
    num_values = int(meta.get("num_values", 0)) if isinstance(meta, dict) else 0
    if num_values <= 0 and dim_index == 0:
        num_values = int(metadata.get("num_cameras", 0))
    if num_values <= 0:
        raise ValueError(f"SIE dim_index {dim_index} has no inferred cardinality in dataset metadata.")
    sie_cfg["dim_index"] = dim_index
    sie_cfg["num_values"] = num_values
    return sie_cfg


def _sharing_config(cfg: Dict[str, Any]) -> tuple[bool, bool]:
    """Return trunk sharing and the fixed shared auxiliary-SIE policy."""
    sharing = cfg["model"]["encoder"].get("sharing", {})
    if "sie" in sharing:
        raise ValueError(
            "model.encoder.sharing.sie was removed; modality SIE now follows sharing.backbone automatically."
        )
    return (
        bool(sharing.get("backbone", True)),
        True,
    )


def _wrap_image_model(
    model: nn.Module,
    cfg: Dict[str, Any],
    *,
    num_modalities: int,
    share_backbone: bool,
    share_sie: bool,
    input_size,
    vit_stem,
    sie_cfg,
) -> nn.Module:
    mcfg = cfg["model"]["encoder"]["backbone"]
    if not share_backbone and num_modalities > 1:
        encoders = [
            ImageBackboneAdapter(
                copy.deepcopy(model),
                input_size=input_size,
                feature_dim=mcfg.get("feature_dim"),
                last_stride=mcfg.get("last_stride", 1),
                vit_stem=vit_stem,
                sie_cfg=sie_cfg,
                num_modalities=(num_modalities if share_sie else 1),
                sie_shared=True,
            )
            for _ in range(num_modalities)
        ]
        if share_sie and encoders:
            shared_sie = encoders[0].sie
            for encoder in encoders[1:]:
                encoder.sie = shared_sie
        return ModalitySpecificBackboneAdapter(encoders, local_modality_ids=not share_sie)
    return ImageBackboneAdapter(
        model,
        input_size=input_size,
        feature_dim=mcfg.get("feature_dim"),
        last_stride=mcfg.get("last_stride", 1),
        vit_stem=vit_stem,
        sie_cfg=sie_cfg,
        num_modalities=num_modalities,
        sie_shared=share_sie,
    )


def _wrap_clip_visual(
    visual: nn.Module,
    cfg: Dict[str, Any],
    *,
    num_modalities: int,
    share_backbone: bool,
    share_sie: bool,
    input_size,
    vit_stem,
    sie_cfg,
) -> nn.Module:
    mcfg = cfg["model"]["encoder"]["backbone"]
    if not share_backbone and num_modalities > 1:
        encoders = [
            CLIPVisionBackboneAdapter(
                copy.deepcopy(visual),
                input_size=input_size,
                feature_dim=mcfg.get("feature_dim"),
                vit_stem=vit_stem,
                sie_cfg=sie_cfg,
                num_modalities=(num_modalities if share_sie else 1),
                sie_shared=True,
            )
            for _ in range(num_modalities)
        ]
        if share_sie and encoders:
            shared_sie = encoders[0].sie
            for encoder in encoders[1:]:
                encoder.sie = shared_sie
        return ModalitySpecificBackboneAdapter(encoders, local_modality_ids=not share_sie)
    return CLIPVisionBackboneAdapter(
        visual,
        input_size=input_size,
        feature_dim=mcfg.get("feature_dim"),
        vit_stem=vit_stem,
        sie_cfg=sie_cfg,
        num_modalities=num_modalities,
        sie_shared=share_sie,
    )


def _wrap_dino_model(
    model: nn.Module,
    cfg: Dict[str, Any],
    *,
    version: str,
    num_modalities: int,
    share_backbone: bool,
    share_sie: bool,
    input_size,
    sie_cfg,
) -> nn.Module:
    mcfg = cfg["model"]["encoder"]["backbone"]
    kwargs = dict(
        version=version,
        input_size=input_size,
        feature_dim=mcfg.get("feature_dim"),
        memory_cfg=mcfg.get("memory", {}),
        sie_cfg=sie_cfg,
        num_modalities=(num_modalities if share_sie else 1),
        sie_shared=True,
    )
    if not share_backbone and num_modalities > 1:
        encoders = [DinoVisionBackboneAdapter(copy.deepcopy(model), **kwargs) for _ in range(num_modalities)]
        if share_sie and encoders:
            shared_sie = encoders[0].sie
            for encoder in encoders[1:]:
                encoder.sie = shared_sie
        return ModalitySpecificBackboneAdapter(encoders, local_modality_ids=not share_sie)
    return DinoVisionBackboneAdapter(model, **kwargs)


def build_backbone(cfg: Dict[str, Any]) -> Tuple[nn.Module, str]:
    """
    Returns (encoder, family) where family in {"cnn", "vit"} for ID neck choice.
    Encoder.forward(x) -> BackboneOutput (single modality batch x [B,3,H,W]).
    """
    mcfg = cfg["model"]["encoder"]["backbone"]
    source = mcfg.get("source", "auto").lower()
    pretrained_value = mcfg.get("pretrained", True)
    pretrained = bool(pretrained_value)
    pretrained_weights_path = resolve_runtime_path(
        mcfg.get("pretrained_weights_path", ""),
        cfg.get("_path_base"),
    )
    input_size = resolve_padded_input_size(cfg)
    vit_stem = resolve_vit_stem_config(cfg)
    sie_cfg = resolve_sie_config(cfg)
    num_modalities = resolve_num_modalities(cfg)
    share_backbone, share_sie = _sharing_config(cfg)

    if source in {"auto", "torchvision"}:
        # When pretrained=false, ignore pretrained_weights_path (random init only).
        pwp = pretrained_weights_path if pretrained else ""
        # When a custom weights file is provided, skip source-library download.
        use_source_pretrained = pretrained and not bool(pwp)
        try:
            model = build_torchvision_model(mcfg["name"], pretrained=use_source_pretrained)
        except ValueError:
            if source != "auto" or not mcfg.get("repo"):
                raise
            model = build_torch_hub_model(mcfg["repo"], mcfg["name"], pretrained=use_source_pretrained, kwargs=mcfg.get("kwargs", {}))
        if not pretrained:
            apply_model_init(model, "kaiming")
        weights_url = mcfg.get("weights_url", "") if pretrained else ""
        if weights_url:
            load_state_dict_from_path_or_url(model, weights_url)
        if pwp:
            load_state_dict_from_path_or_url(model, pwp)
        net = _wrap_image_model(
            model,
            cfg,
            num_modalities=num_modalities,
            share_backbone=share_backbone,
            share_sie=share_sie,
            input_size=input_size,
            vit_stem=vit_stem,
            sie_cfg=sie_cfg,
        )
        return net, infer_backbone_family(mcfg["name"], mcfg.get("family"))

    if source in {"torch", "torch_hub", "hub"}:
        repo = mcfg.get("repo", "")
        if not repo:
            raise ValueError("torch_hub backbone requires model.encoder.backbone.repo.")
        pwp = pretrained_weights_path if pretrained else ""
        use_source_pretrained = pretrained and not bool(pwp)
        model = build_torch_hub_model(repo, mcfg["name"], pretrained=use_source_pretrained, kwargs=mcfg.get("kwargs", {}))
        if not pretrained:
            apply_model_init(model, "kaiming")
        weights_url = mcfg.get("weights_url", "") if pretrained else ""
        if weights_url:
            load_state_dict_from_path_or_url(model, weights_url)
        if pwp:
            load_state_dict_from_path_or_url(model, pwp)
        net = _wrap_image_model(
            model,
            cfg,
            num_modalities=num_modalities,
            share_backbone=share_backbone,
            share_sie=share_sie,
            input_size=input_size,
            vit_stem=vit_stem,
            sie_cfg=sie_cfg,
        )
        return net, infer_backbone_family(mcfg["name"], mcfg.get("family"))

    if source in {"dinov2", "dino_v2", "dinov3", "dino_v3"}:
        version = "dinov3" if "3" in source else "dinov2"
        if vit_stem is not None and bool(vit_stem.get("enabled", False)):
            print("Warning: vit_stem is ignored for official DINO backbones.")
        if version == "dinov2" and bool(mcfg.get("memory", {}).get("enabled", False)):
            raise ValueError("DINO register-token Memory mode is supported only for DINOv3.")
        repo = str(mcfg.get("repo") or f"facebookresearch/{version}")
        hub_kwargs = dict(mcfg.get("kwargs", {}))
        repo_source = str(mcfg.get("repo_source", "github")).strip().lower()
        if repo_source not in {"github", "local"}:
            raise ValueError("DINO backbone repo_source must be 'github' or 'local'.")
        if repo_source == "local":
            repo = resolve_runtime_path(repo, cfg.get("_path_base"))
            hub_kwargs["source"] = "local"
        weights_url = str(mcfg.get("weights_url", "")) if pretrained else ""
        pwp = pretrained_weights_path if pretrained else ""
        weights_source = pwp or weights_url
        if version == "dinov3" and pretrained:
            if not weights_source:
                raise ValueError(
                    "Pretrained DINOv3 requires backbone.pretrained_weights_path or weights_url "
                    "from the official gated model download."
                )
            hub_kwargs["weights"] = weights_source
        use_source_pretrained = pretrained and (version == "dinov3" or not bool(pwp))
        model = build_torch_hub_model(repo, mcfg["name"], pretrained=use_source_pretrained, kwargs=hub_kwargs)
        if not pretrained:
            apply_model_init(model, "kaiming")
        elif version == "dinov2":
            if weights_url:
                load_state_dict_from_path_or_url(model, weights_url)
            if pwp:
                load_state_dict_from_path_or_url(model, pwp)
        net = _wrap_dino_model(
            model,
            cfg,
            version=version,
            num_modalities=num_modalities,
            share_backbone=share_backbone,
            share_sie=share_sie,
            input_size=input_size,
            sie_cfg=sie_cfg,
        )
        return net, "vit"

    if source == "timm":
        pwp = pretrained_weights_path if pretrained else ""
        use_source_pretrained = pretrained and not bool(pwp)
        model = build_timm_model(mcfg["name"], pretrained=use_source_pretrained, kwargs=mcfg.get("kwargs", {}))
        if not pretrained:
            apply_model_init(model, "kaiming")
        weights_url = mcfg.get("weights_url", "") if pretrained else ""
        if weights_url:
            load_state_dict_from_path_or_url(model, weights_url)
        if pwp:
            load_state_dict_from_path_or_url(model, pwp)
        net = _wrap_image_model(
            model,
            cfg,
            num_modalities=num_modalities,
            share_backbone=share_backbone,
            share_sie=share_sie,
            input_size=input_size,
            vit_stem=vit_stem,
            sie_cfg=sie_cfg,
        )
        return net, infer_backbone_family(mcfg["name"], mcfg.get("family"))

    if source in {"huggingface-hub", "huggingface", "transformers", "hf"}:
        if vit_stem is not None and bool(vit_stem.get("enabled", False)):
            print("Warning: vit_stem is ignored for HuggingFace/Transformers backbones.")
        if sie_cfg is not None:
            print("Warning: SIE is ignored for HuggingFace/Transformers backbones.")
        pwp = pretrained_weights_path if pretrained else ""
        # When a custom weights file is provided, build from config (no download).
        if pwp and pretrained_value is not False:
            hf_pretrained = False
        else:
            hf_pretrained = pretrained_value
        model = build_huggingface_model(mcfg["name"], pretrained=hf_pretrained, kwargs=mcfg.get("kwargs", {}))
        if not pretrained:
            apply_model_init(model, "kaiming")
        weights_url = mcfg.get("weights_url", "") if pretrained else ""
        if weights_url:
            load_state_dict_from_path_or_url(model, weights_url)
        if pwp:
            load_state_dict_from_path_or_url(model, pwp)
        net = HuggingFaceVisionBackboneAdapter(model, input_size=input_size)
        return net, infer_backbone_family(mcfg["name"], mcfg.get("family"))

    if source in {"openai-clip", "clip"}:
        weights_url = mcfg.get("weights_url", "") if pretrained else ""
        pwp = pretrained_weights_path if pretrained else ""
        weights_source = pwp or weights_url or None
        visual = build_openai_clip_vision_model(
            mcfg["name"],
            pretrained=pretrained_value,
            kwargs=mcfg.get("kwargs", {}),
            weights_source=weights_source,
        )
        if not pretrained:
            apply_model_init(visual, "kaiming")
        net = _wrap_clip_visual(
            visual,
            cfg,
            num_modalities=num_modalities,
            share_backbone=share_backbone,
            share_sie=share_sie,
            input_size=input_size,
            vit_stem=vit_stem,
            sie_cfg=sie_cfg,
        )
        return net, "vit"

    if source in {"open-clip", "open_clip"}:
        weights_url = mcfg.get("weights_url", "") if pretrained else ""
        pwp = pretrained_weights_path if pretrained else ""
        weights_source = pwp or weights_url or None
        visual = build_open_clip_vision_model(
            mcfg["name"],
            pretrained=pretrained_value,
            kwargs=mcfg.get("kwargs", {}),
            weights_source=weights_source,
        )
        if not pretrained:
            apply_model_init(visual, "kaiming")
        net = _wrap_clip_visual(
            visual,
            cfg,
            num_modalities=num_modalities,
            share_backbone=share_backbone,
            share_sie=share_sie,
            input_size=input_size,
            vit_stem=vit_stem,
            sie_cfg=sie_cfg,
        )
        return net, "vit"

    if source == "custom":
        target = mcfg.get("target", "")
        model = build_custom_model(target, mcfg)
        if not pretrained:
            apply_model_init(model, "kaiming")
        weights_url = mcfg.get("weights_url", "") if pretrained else ""
        pwp = pretrained_weights_path if pretrained else ""
        if weights_url:
            load_state_dict_from_path_or_url(model, weights_url)
        if pwp:
            load_state_dict_from_path_or_url(model, pwp)
        net = _wrap_image_model(
            model,
            cfg,
            num_modalities=num_modalities,
            share_backbone=share_backbone,
            share_sie=share_sie,
            input_size=input_size,
            vit_stem=vit_stem,
            sie_cfg=sie_cfg,
        )
        return net, infer_backbone_family(mcfg["name"], mcfg.get("family"))

    raise ValueError(
        f"Unknown backbone source: {source!r}. "
        "Use auto, torchvision, timm, torch_hub, dinov2, dinov3, openai-clip, open-clip, "
        "huggingface-hub, transformers, or custom."
    )
