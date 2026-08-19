"""Orthogonality and MV-SRBF auxiliary losses."""
from __future__ import annotations

import torch


def ortho_intra_loss(spatial_tokens: torch.Tensor) -> torch.Tensor:
    """
    spatial_tokens: [B, N, D] patch tokens (exclude CLS).
    Mean over batch of sum_{i!=j} |cos(t_i,t_j)|.
    """
    b, n, d = spatial_tokens.shape
    if n < 2:
        return spatial_tokens.new_zeros(())
    x = spatial_tokens
    x = x / (x.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    sim = torch.bmm(x, x.transpose(1, 2))
    eye = torch.eye(n, device=x.device, dtype=x.dtype).unsqueeze(0).expand(b, -1, -1)
    off = (1.0 - eye) * sim.abs()
    return off.sum(dim=(1, 2)).mean() / (n * (n - 1))


def ortho_inter_loss(feats: torch.Tensor) -> torch.Tensor:
    """
    feats: [B, M, D] modality vectors, L2-normalized cosine orthogonality between modalities.
    """
    b, m, d = feats.shape
    if m < 2:
        return feats.new_zeros(())
    x = feats / (feats.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    sim = torch.bmm(x, x.transpose(1, 2))
    triu = torch.triu_indices(m, m, offset=1, device=feats.device)
    return sim[:, triu[0], triu[1]].abs().mean()
