"""Configurable inference feature extraction for multi-modal ReID models."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn

from evaluation.modality_utils import availability_to_cls_mask


def _mean_fill_missing(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill missing [B,M,D] slots with each sample's available-modality mean."""
    available = (~mask).to(features.dtype).unsqueeze(-1)
    mean = (features * available).sum(dim=1, keepdim=True) / available.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)
    return torch.where(mask.unsqueeze(-1), mean.expand_as(features), features)


def _aggregate_stage(
    model: nn.Module,
    stage_features: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    *,
    stage: str,
    aggregation: str,
    neck_feat: str,
) -> torch.Tensor:
    if aggregation not in {"mean", "concat", "fusion"}:
        raise ValueError("feature_aggregation must be mean, concat, or fusion.")

    features = stage_features[neck_feat]
    completed = _mean_fill_missing(features, mask) if stage == "modal" else features
    if aggregation == "mean":
        if stage == "modal":
            available = (~mask).to(features.dtype).unsqueeze(-1)
            return (features * available).sum(dim=1) / available.sum(dim=1).clamp_min(1.0)
        return completed.mean(dim=1)
    if aggregation == "concat":
        return completed.flatten(1)
    fusion_input = stage_features.get("fusion_input", stage_features["after"])
    if stage == "modal":
        fusion_input = _mean_fill_missing(fusion_input, mask)
    return model.fuse_inference_features(fusion_input, neck_feat=neck_feat)


@torch.no_grad()
def extract_fused_feature(
    model: nn.Module,
    modalities: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    aux_indices: Dict[int, torch.Tensor] | None = None,
    neck_feat: str = "before",
    feature_stage: str = "fusion",
    feature_aggregation: str = "mean",
) -> torch.Tensor:
    """Extract modal, recovered, or native fusion descriptors.

    feature_aggregation applies to modal and recover. The fusion stage always
    returns the model's configured fusion output.
    """
    model.eval()
    stage = str(feature_stage).strip().lower()
    aggregation = str(feature_aggregation).strip().lower()
    neck = str(neck_feat).strip().lower()
    aliases = {"single": "modal", "recovered": "recover", "fused": "fusion", "final": "fusion"}
    stage = aliases.get(stage, stage)
    if stage not in {"modal", "recover", "fusion"}:
        raise ValueError("feature_stage must be modal, recover, or fusion.")
    if neck not in {"before", "after"}:
        raise ValueError("neck_feat must be before or after.")
    if stage == "recover" and not bool(getattr(model, "recovery_enabled", False)):
        raise ValueError("feature_stage=recover requires the model recovery branch to be enabled.")

    out = model(
        modalities,
        labels,
        mask=mask,
        aux_indices=aux_indices,
        return_loss_dict=False,
        use_recovery=stage in {"recover", "fusion"},
        use_fusion=stage == "fusion",
    )
    if stage == "fusion":
        return out["fuse_feat_after" if neck == "after" else "fuse_feat_pre"]

    stage_features = out["stage_features"][stage]
    return _aggregate_stage(
        model,
        stage_features,
        mask,
        stage=stage,
        aggregation=aggregation,
        neck_feat=neck,
    )


def extract_fused_feature_for_availability(
    model: nn.Module,
    modalities: Dict[str, torch.Tensor],
    modal_keys: Sequence[str],
    available_indices: Sequence[int],
    *,
    aux_indices: Dict[int, torch.Tensor] | None = None,
    neck_feat: str = "before",
    feature_stage: str = "fusion",
    feature_aggregation: str = "mean",
) -> torch.Tensor:
    """Extract descriptors when a batch shares one availability pattern."""
    batch_size = modalities[modal_keys[0]].shape[0]
    device = modalities[modal_keys[0]].device
    mask = availability_to_cls_mask(
        batch_size,
        len(modal_keys),
        available_indices,
        device,
    )
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    return extract_fused_feature(
        model,
        modalities,
        labels,
        mask=mask,
        aux_indices=aux_indices,
        neck_feat=neck_feat,
        feature_stage=feature_stage,
        feature_aggregation=feature_aggregation,
    )
