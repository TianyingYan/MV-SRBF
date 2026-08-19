"""Configurable ReID classification, metric-learning, and center losses."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def label_smoothing_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 0.1,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    n_class = logits.size(1)
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(epsilon / max(1, n_class - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - epsilon)
    return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
    """Cross entropy with optional label smoothing."""
    if epsilon > 0:
        return label_smoothing_cross_entropy(logits, targets, epsilon)
    return F.cross_entropy(logits, targets)


def _normalized_logits(features: torch.Tensor, classifier_weight: torch.Tensor, scale: float) -> torch.Tensor:
    cosine = F.linear(F.normalize(features), F.normalize(classifier_weight))
    return float(scale) * cosine


def _margin_softmax_logits(
    features: torch.Tensor,
    classifier_weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_type: str,
    scale: float,
    margin: float,
    easy_margin: bool = False,
    sphere_m: int = 4,
) -> torch.Tensor:
    cosine = F.linear(F.normalize(features), F.normalize(classifier_weight)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    one_hot = F.one_hot(targets, num_classes=cosine.shape[1]).to(dtype=cosine.dtype, device=cosine.device)
    if loss_type in {"arcface", "arc"}:
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(0.0))
        phi = cosine * torch.cos(cosine.new_tensor(margin)) - sine * torch.sin(cosine.new_tensor(margin))
        if easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            threshold = torch.cos(cosine.new_tensor(torch.pi - margin))
            mm = torch.sin(cosine.new_tensor(torch.pi - margin)) * margin
            phi = torch.where(cosine > threshold, phi, cosine - mm)
    elif loss_type in {"cosface", "am_softmax", "am"}:
        phi = cosine - margin
    elif loss_type in {"sphereface", "sphere"}:
        theta = torch.acos(cosine)
        phi = torch.cos(int(sphere_m) * theta)
    else:
        raise ValueError(f"Unknown margin-softmax CE type: {loss_type!r}.")
    return scale * (one_hot * phi + (1.0 - one_hot) * cosine)


def classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    cfg: Dict[str, Any] | None = None,
    *,
    features: torch.Tensor | None = None,
    classifier_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch classifier losses from YAML config."""
    cfg = cfg or {}
    loss_type = str(cfg.get("name", cfg.get("type", "ce"))).lower()
    epsilon = float(cfg.get("epsilon", 0.1)) if bool(cfg.get("label_smooth", True)) else 0.0
    if loss_type in {"ce", "cross_entropy", "softmax", "label_smooth", "labelsmooth"}:
        return cross_entropy_loss(logits, targets, epsilon)
    if loss_type in {"normalized_softmax", "norm_softmax", "cosine_softmax", "cosine"}:
        if features is None or classifier_weight is None:
            raise ValueError(f"{loss_type} requires post-neck features and classifier weights.")
        scaled_logits = _normalized_logits(
            features,
            classifier_weight,
            scale=float(cfg.get("s", cfg.get("logit_scale", 30.0))),
        )
        return cross_entropy_loss(scaled_logits, targets, epsilon)
    if loss_type in {"arcface", "arc", "cosface", "am_softmax", "am", "sphereface", "sphere"}:
        if features is None or classifier_weight is None:
            raise ValueError(f"{loss_type} requires post-neck features and classifier weights.")
        margin_logits = _margin_softmax_logits(
            features,
            classifier_weight,
            targets,
            loss_type=loss_type,
            scale=float(cfg.get("s", cfg.get("logit_scale", 30.0))),
            margin=float(cfg.get("margin", 0.5 if loss_type in {"arcface", "arc", "sphereface", "sphere"} else 0.35)),
            easy_margin=bool(cfg.get("easy_margin", False)),
            sphere_m=int(cfg.get("sphere_m", cfg.get("m", 4))),
        )
        return cross_entropy_loss(margin_logits, targets, epsilon)
    raise ValueError(
        f"Unknown CE loss type: {loss_type!r}. "
        "Use ce, label_smooth, normalized_softmax, arcface, cosface, am_softmax, or sphereface."
    )


def _pairwise_dist(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.cdist(embeddings, embeddings, p=2)


def _pairwise_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    return F.normalize(embeddings, dim=1) @ F.normalize(embeddings, dim=1).t()


def _label_pair_masks(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
    pos_mask = labels_equal & ~eye
    neg_mask = ~labels_equal
    return pos_mask, neg_mask


def _zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def _batch_hard_triplet_from_masks(
    embeddings: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    *,
    margin: float,
    soft: bool,
    temperature: float,
) -> torch.Tensor:
    dist = _pairwise_dist(embeddings)
    losses = []
    for i in range(dist.size(0)):
        if not pos_mask[i].any() or not neg_mask[i].any():
            continue
        delta = dist[i][pos_mask[i]].max() - dist[i][neg_mask[i]].min() + float(margin)
        if soft:
            losses.append(F.softplus(delta / max(float(temperature), 1e-12)))
        else:
            losses.append(F.relu(delta))
    if not losses:
        return _zero_like_loss(embeddings)
    return torch.stack(losses).mean()


def _batch_all_triplet_from_masks(
    embeddings: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    *,
    margin: float,
    softmax_style: bool,
    temperature: float,
) -> torch.Tensor:
    dist = _pairwise_dist(embeddings)
    valid = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
    if not valid.any():
        return _zero_like_loss(embeddings)
    if softmax_style:
        temp = max(float(temperature), 1e-12)
        pos_logits = (-(dist.unsqueeze(2) + float(margin)) / temp).expand_as(valid)
        neg_logits = (-(dist.unsqueeze(1)) / temp).expand_as(valid)
        logits = torch.stack([pos_logits[valid], neg_logits[valid]], dim=-1)
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=embeddings.device)
        return F.cross_entropy(logits, targets)
    losses = F.relu(dist.unsqueeze(2) - dist.unsqueeze(1) + float(margin))
    return losses[valid].mean()


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """Classic ReID batch-hard triplet with hinge margin."""
    pos_mask, neg_mask = _label_pair_masks(labels)
    return _batch_hard_triplet_from_masks(
        embeddings,
        pos_mask,
        neg_mask,
        margin=margin,
        soft=False,
        temperature=1.0,
    )


def batch_all_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """Classic all-triplet hinge loss over valid anchor-positive-negative tuples."""
    pos_mask, neg_mask = _label_pair_masks(labels)
    return _batch_all_triplet_from_masks(
        embeddings,
        pos_mask,
        neg_mask,
        margin=margin,
        softmax_style=False,
        temperature=1.0,
    )


def batch_hard_softmax_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Softplus batch-hard triplet, often called hard softmax triplet in ReID code."""
    pos_mask, neg_mask = _label_pair_masks(labels)
    return _batch_hard_triplet_from_masks(
        embeddings,
        pos_mask,
        neg_mask,
        margin=margin,
        soft=True,
        temperature=temperature,
    )


def batch_all_softmax_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Cross-entropy over every valid positive/negative distance pair."""
    pos_mask, neg_mask = _label_pair_masks(labels)
    return _batch_all_triplet_from_masks(
        embeddings,
        pos_mask,
        neg_mask,
        margin=margin,
        softmax_style=True,
        temperature=temperature,
    )


def contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Pairwise contrastive loss over unordered same-ID and different-ID pairs."""
    pos_mask, neg_mask = _label_pair_masks(labels)
    return _contrastive_from_masks(embeddings, pos_mask, neg_mask, margin=margin)


def _contrastive_from_masks(
    embeddings: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    dist = _pairwise_dist(embeddings)
    upper = torch.triu(torch.ones_like(pos_mask, dtype=torch.bool), diagonal=1)
    pos = pos_mask & upper
    neg = neg_mask & upper
    terms = []
    if pos.any():
        terms.append(dist[pos].pow(2))
    if neg.any():
        terms.append(F.relu(float(margin) - dist[neg]).pow(2))
    if not terms:
        return _zero_like_loss(embeddings)
    return torch.cat(terms).mean()


def circle_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.25,
    gamma: float = 256.0,
) -> torch.Tensor:
    """Pair-similarity Circle loss with same-ID positives."""
    pos_mask, neg_mask = _label_pair_masks(labels)
    return _circle_from_masks(embeddings, pos_mask, neg_mask, margin=margin, gamma=gamma)


def _circle_from_masks(
    embeddings: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    *,
    margin: float,
    gamma: float,
) -> torch.Tensor:
    sim = _pairwise_similarity(embeddings)
    losses = []
    m = float(margin)
    g = float(gamma)
    delta_p = 1.0 - m
    delta_n = m
    for i in range(sim.size(0)):
        sp = sim[i][pos_mask[i]]
        sn = sim[i][neg_mask[i]]
        if sp.numel() == 0 or sn.numel() == 0:
            continue
        ap = torch.clamp_min(-sp.detach() + 1.0 + m, 0.0)
        an = torch.clamp_min(sn.detach() + m, 0.0)
        logit_p = -g * ap * (sp - delta_p)
        logit_n = g * an * (sn - delta_n)
        losses.append(F.softplus(torch.logsumexp(logit_p, dim=0) + torch.logsumexp(logit_n, dim=0)))
    if not losses:
        return _zero_like_loss(embeddings)
    return torch.stack(losses).mean()


def multi_similarity_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    scale_pos: float = 2.0,
    scale_neg: float = 40.0,
    base: float = 0.5,
    epsilon: float = 0.1,
) -> torch.Tensor:
    """Multi-Similarity loss from pair similarities."""
    pos_mask, neg_mask = _label_pair_masks(labels)
    return _multi_similarity_from_masks(
        embeddings,
        pos_mask,
        neg_mask,
        scale_pos=scale_pos,
        scale_neg=scale_neg,
        base=base,
        epsilon=epsilon,
    )


def _multi_similarity_from_masks(
    embeddings: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    *,
    scale_pos: float,
    scale_neg: float,
    base: float,
    epsilon: float,
) -> torch.Tensor:
    sim = _pairwise_similarity(embeddings)
    losses = []
    for i in range(sim.size(0)):
        pos_pair = sim[i][pos_mask[i]]
        neg_pair = sim[i][neg_mask[i]]
        if pos_pair.numel() == 0 or neg_pair.numel() == 0:
            continue
        pos_pair = pos_pair[pos_pair < neg_pair.max() + float(epsilon)]
        neg_pair = neg_pair[neg_pair > pos_pair.min() - float(epsilon)] if pos_pair.numel() > 0 else neg_pair
        if pos_pair.numel() == 0 or neg_pair.numel() == 0:
            continue
        pos_loss = torch.log1p(torch.exp(-float(scale_pos) * (pos_pair - float(base))).sum()) / float(scale_pos)
        neg_loss = torch.log1p(torch.exp(float(scale_neg) * (neg_pair - float(base))).sum()) / float(scale_neg)
        losses.append(pos_loss + neg_loss)
    if not losses:
        return _zero_like_loss(embeddings)
    return torch.stack(losses).mean()


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss using same-ID positives in the mini-batch."""
    pos_mask, _ = _label_pair_masks(labels)
    return _supervised_contrastive_from_masks(embeddings, pos_mask, temperature=temperature)


def _supervised_contrastive_from_masks(
    embeddings: torch.Tensor,
    pos_mask: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    sim = _pairwise_similarity(embeddings) / max(float(temperature), 1e-12)
    eye = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    logits = sim.masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if not valid.any():
        return _zero_like_loss(embeddings)
    loss = -(log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1)[valid] / pos_count[valid].clamp_min(1))
    return loss.mean()


def _metric_variant_from_cfg(cfg: Dict[str, Any]) -> str:
    if "name" not in cfg and "type" not in cfg:
        raise ValueError("Metric loss config requires `name`. Use hard, all, hard_softmax, all_softmax, or another supported metric loss.")
    return str(cfg.get("name", cfg.get("type"))).lower()


def metric_learning_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    cfg: Dict[str, Any] | None = None,
) -> torch.Tensor:
    """Dispatch ReID metric-learning variants from YAML config."""
    cfg = cfg or {}
    variant = _metric_variant_from_cfg(cfg)
    margin = float(cfg.get("margin", 0.0 if "softmax" in variant or "soft" in variant else 0.3))
    temperature = float(cfg.get("temperature", cfg.get("temp", 1.0)))
    if variant in {"hard", "batch_hard", "classic_hard", "margin_hard"}:
        return batch_hard_triplet_loss(embeddings, labels, margin)
    if variant in {"all", "batch_all", "classic_all", "margin_all"}:
        return batch_all_triplet_loss(embeddings, labels, margin)
    if variant in {"hard_soft", "hard_softmax", "soft_hard", "softplus_hard"}:
        return batch_hard_softmax_triplet_loss(embeddings, labels, margin, temperature)
    if variant in {"all_soft", "all_softmax", "soft_all"}:
        return batch_all_softmax_triplet_loss(embeddings, labels, margin, temperature)
    if variant in {"contrastive", "pair_contrastive"}:
        return contrastive_loss(embeddings, labels, margin=float(cfg.get("margin", 1.0)))
    if variant in {"circle", "circle_loss"}:
        return circle_loss(
            embeddings,
            labels,
            margin=float(cfg.get("margin", 0.25)),
            gamma=float(cfg.get("gamma", cfg.get("scale", 256.0))),
        )
    if variant in {"multi_similarity", "multisimilarity", "ms"}:
        return multi_similarity_loss(
            embeddings,
            labels,
            scale_pos=float(cfg.get("scale_pos", cfg.get("alpha", 2.0))),
            scale_neg=float(cfg.get("scale_neg", cfg.get("beta", 40.0))),
            base=float(cfg.get("base", cfg.get("lambda", 0.5))),
            epsilon=float(cfg.get("epsilon", 0.1)),
        )
    if variant in {"supcon", "supervised_contrastive", "supervised_contrast"}:
        return supervised_contrastive_loss(embeddings, labels, temperature=temperature)
    raise ValueError(
        f"Unknown metric loss type: {variant!r}. "
        "Use hard, all, hard_softmax, all_softmax, contrastive, circle, multi_similarity, or supcon."
    )


def cross_modal_triplet_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    cfg: Dict[str, Any] | None = None,
) -> torch.Tensor:
    """Cross-modal metric loss whose positives are same-ID and different-modality."""
    cfg = cfg or {}
    if features.ndim != 3:
        raise ValueError(f"Expected features shaped [B, M, D], got {tuple(features.shape)}.")
    b, m, d = features.shape
    if m <= 1:
        return _zero_like_loss(features)
    flat = features.reshape(b * m, d)
    flat_labels = labels.view(b, 1).expand(b, m).reshape(-1)
    modal_ids = torch.arange(m, device=features.device).view(1, m).expand(b, m).reshape(-1)
    same_id = flat_labels.unsqueeze(0) == flat_labels.unsqueeze(1)
    diff_modal = modal_ids.unsqueeze(0) != modal_ids.unsqueeze(1)
    eye = torch.eye(flat.shape[0], dtype=torch.bool, device=features.device)
    pos_mask = same_id & diff_modal & ~eye
    neg_mask = ~same_id
    variant = _metric_variant_from_cfg(cfg)
    margin = float(cfg.get("margin", 0.0 if "softmax" in variant or "soft" in variant else 0.3))
    temperature = float(cfg.get("temperature", cfg.get("temp", 1.0)))
    if variant in {"hard", "batch_hard", "classic_hard", "margin_hard"}:
        return _batch_hard_triplet_from_masks(
            flat,
            pos_mask,
            neg_mask,
            margin=margin,
            soft=False,
            temperature=temperature,
        )
    if variant in {"all", "batch_all", "classic_all", "margin_all"}:
        return _batch_all_triplet_from_masks(
            flat,
            pos_mask,
            neg_mask,
            margin=margin,
            softmax_style=False,
            temperature=temperature,
        )
    if variant in {"hard_soft", "hard_softmax", "soft_hard", "softplus_hard"}:
        return _batch_hard_triplet_from_masks(
            flat,
            pos_mask,
            neg_mask,
            margin=margin,
            soft=True,
            temperature=temperature,
        )
    if variant in {"all_soft", "all_softmax", "soft_all"}:
        return _batch_all_triplet_from_masks(
            flat,
            pos_mask,
            neg_mask,
            margin=margin,
            softmax_style=True,
            temperature=temperature,
        )
    if variant in {"contrastive", "pair_contrastive"}:
        return _contrastive_from_masks(flat, pos_mask, neg_mask, margin=float(cfg.get("margin", 1.0)))
    if variant in {"circle", "circle_loss"}:
        return _circle_from_masks(
            flat,
            pos_mask,
            neg_mask,
            margin=float(cfg.get("margin", 0.25)),
            gamma=float(cfg.get("gamma", cfg.get("scale", 256.0))),
        )
    if variant in {"multi_similarity", "multisimilarity", "ms"}:
        return _multi_similarity_from_masks(
            flat,
            pos_mask,
            neg_mask,
            scale_pos=float(cfg.get("scale_pos", cfg.get("alpha", 2.0))),
            scale_neg=float(cfg.get("scale_neg", cfg.get("beta", 40.0))),
            base=float(cfg.get("base", cfg.get("lambda", 0.5))),
            epsilon=float(cfg.get("epsilon", 0.1)),
        )
    if variant in {"supcon", "supervised_contrastive", "supervised_contrast"}:
        return _supervised_contrastive_from_masks(flat, pos_mask, temperature=temperature)
    raise ValueError(f"Unsupported cross-modal metric loss type: {variant!r}.")


class CenterLoss(nn.Module):
    """Class centers c_j in feature space; minimize sum ||x_i - c_y||^2 / (2B)."""

    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        centers = self.centers.index_select(0, labels)
        return ((x - centers) ** 2).sum() / (2.0 * x.size(0))
