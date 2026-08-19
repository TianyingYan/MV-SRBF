"""Configurable ReID norm neck; classifier uses post-neck features."""
from __future__ import annotations

import torch
import torch.nn as nn

from utils.weight_init import init_classifier, init_kaiming


def build_id_neck(family: str, dim: int) -> nn.Module:
    return nn.BatchNorm1d(dim)


class ModalReIDHead(nn.Module):
    """Norm neck + linear classifier. Triplet/Center use pre-neck features outside this module."""

    def __init__(
        self,
        family: str,
        dim: int,
        num_classes: int,
        with_neck: bool = True,
    ):
        super().__init__()
        self.neck = build_id_neck(family, dim) if with_neck else nn.Identity()
        self.classifier = nn.Linear(dim, num_classes, bias=False)
        self.neck.apply(init_kaiming)
        self.classifier.apply(init_classifier)

    def forward(self, feat_pre: torch.Tensor):
        feat_id = self.neck(feat_pre)
        logits = self.classifier(feat_id)
        return logits, feat_id, feat_pre
