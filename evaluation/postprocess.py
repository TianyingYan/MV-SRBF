"""Feature post-processing for ReID inference."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.metrics import pairwise_distance


def maybe_normalize_features(feat: torch.Tensor, enabled: bool) -> torch.Tensor:
    """L2-normalize features when requested by inference.feat_norm."""
    return F.normalize(feat, dim=1) if enabled else feat


def apply_aqe(
    q_feat: torch.Tensor,
    g_feat: torch.Tensor,
    *,
    metric: str,
    alpha: float = 3.0,
    qe_time: int = 1,
    qe_k: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average query/gallery features with nearest neighbors for alpha query expansion."""
    q = q_feat
    g = g_feat
    k = max(1, int(qe_k))
    for _ in range(max(0, int(qe_time))):
        dist_qg = pairwise_distance(q, g, metric)
        dist_gg = pairwise_distance(g, g, metric)
        q_top = dist_qg.topk(min(k, g.shape[0]), largest=False).indices
        g_top = dist_gg.topk(min(k + 1, g.shape[0]), largest=False).indices
        q_weights = torch.linspace(float(alpha), 1.0, q_top.shape[1], device=q.device).view(1, -1, 1)
        g_weights = torch.linspace(float(alpha), 1.0, g_top.shape[1], device=g.device).view(1, -1, 1)
        q = torch.cat([q.unsqueeze(1), g[q_top] * q_weights], dim=1).mean(dim=1)
        g = torch.cat([g.unsqueeze(1), g[g_top] * g_weights], dim=1).mean(dim=1)
        q = F.normalize(q, dim=1)
        g = F.normalize(g, dim=1)
    return q, g


def neighbor_overlap_rerank(
    q_feat: torch.Tensor,
    g_feat: torch.Tensor,
    *,
    metric: str,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> torch.Tensor:
    """Blend original distance with neighbor-overlap distance using k2-expanded neighbor sets."""
    original = pairwise_distance(q_feat, g_feat, metric)
    all_feat = torch.cat([q_feat, g_feat], dim=0)
    all_dist = pairwise_distance(all_feat, all_feat, metric)
    k = min(max(1, int(k1)), all_feat.shape[0])
    k_expand = min(max(1, int(k2)), all_feat.shape[0])
    neighbors = all_dist.topk(k, largest=False).indices.cpu().numpy()
    expand_neighbors = all_dist.topk(k_expand, largest=False).indices.cpu().numpy()
    qn = neighbors[: q_feat.shape[0]]
    gn = neighbors[q_feat.shape[0] :]
    jaccard = np.zeros((q_feat.shape[0], g_feat.shape[0]), dtype=np.float32)

    def expanded_set(row: np.ndarray) -> set[int]:
        values = set(int(x) for x in row)
        if k_expand > 1:
            for idx in row:
                values.update(int(x) for x in expand_neighbors[int(idx)])
        return values

    for qi in range(qn.shape[0]):
        q_set = expanded_set(qn[qi])
        for gi in range(gn.shape[0]):
            g_set = expanded_set(gn[gi])
            inter = len(q_set & g_set)
            union = max(1, len(q_set | g_set))
            jaccard[qi, gi] = 1.0 - float(inter) / float(union)
    jac = torch.from_numpy(jaccard).to(device=original.device, dtype=original.dtype)
    lam = float(lambda_value)
    return lam * original + (1.0 - lam) * jac


def faiss_distance(
    q_feat: torch.Tensor,
    g_feat: torch.Tensor,
    *,
    metric: str,
    use_gpu: bool = False,
    index_type: str = "flat",
) -> torch.Tensor:
    """Return a full query-gallery distance matrix using a Faiss flat index."""
    index_name = str(index_type).lower()
    if index_name != "flat":
        raise ValueError("Only Faiss index='flat' is currently supported for exact ReID evaluation.")

    try:
        import faiss
    except ImportError as exc:
        raise ImportError("Faiss search backend requires `faiss` or `faiss-cpu` to be installed.") from exc

    metric_name = metric.lower()
    q_np = q_feat.detach().float().cpu().numpy()
    g_np = g_feat.detach().float().cpu().numpy()

    if metric_name in {"cosine", "cos"}:
        faiss.normalize_L2(q_np)
        faiss.normalize_L2(g_np)
        index = faiss.IndexFlatIP(g_np.shape[1])
        convert = lambda values: 1.0 - values
    elif metric_name in {"euclidean", "l2"}:
        index = faiss.IndexFlatL2(g_np.shape[1])
        convert = lambda values: np.sqrt(np.maximum(values, 0.0))
    else:
        raise ValueError(f"Unsupported inference metric: {metric!r}. Use cosine/cos or euclidean/l2.")

    if bool(use_gpu):
        try:
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, 0, index)
        except AttributeError:
            raise RuntimeError("Faiss GPU search requested, but this Faiss build does not provide GPU support.")

    index.add(g_np)
    distances, indices = index.search(q_np, g_np.shape[0])
    full = np.empty((q_np.shape[0], g_np.shape[0]), dtype=np.float32)
    rows = np.arange(q_np.shape[0])[:, None]
    full[rows, indices] = convert(distances).astype(np.float32)
    return torch.from_numpy(full).to(device=q_feat.device, dtype=q_feat.dtype)


def search_distance(
    q_feat: torch.Tensor,
    g_feat: torch.Tensor,
    *,
    metric: str,
    search_cfg: Dict[str, Any] | None = None,
) -> torch.Tensor:
    """Compute query-gallery distance with the configured backend."""
    cfg = search_cfg or {}
    backend = str(cfg.get("backend", "torch")).lower()
    if backend == "torch":
        return pairwise_distance(q_feat, g_feat, metric)
    if backend == "faiss":
        faiss_cfg = cfg.get("faiss", {})
        return faiss_distance(
            q_feat,
            g_feat,
            metric=metric,
            use_gpu=bool(faiss_cfg.get("use_gpu", False)),
            index_type=str(faiss_cfg.get("index", "flat")),
        )
    raise ValueError(f"Unsupported inference search backend: {backend!r}. Use torch or faiss.")


def inference_distance(
    q_feat: torch.Tensor,
    g_feat: torch.Tensor,
    metric: str,
    infer_cfg: Dict[str, Any] | None = None,
) -> torch.Tensor:
    """Apply configured AQE/re-ranking and return the final query-gallery distance."""
    cfg = infer_cfg or {}
    q = maybe_normalize_features(q_feat, bool(cfg.get("feat_norm", True)))
    g = maybe_normalize_features(g_feat, bool(cfg.get("feat_norm", True)))

    aqe = cfg.get("aqe", {})
    if aqe.get("enabled", False):
        q, g = apply_aqe(
            q,
            g,
            metric=metric,
            alpha=float(aqe.get("alpha", 3.0)),
            qe_time=int(aqe.get("qe_time", 1)),
            qe_k=int(aqe.get("qe_k", 5)),
        )

    rerank = cfg.get("re_ranking", {})
    if rerank.get("enabled", False):
        return neighbor_overlap_rerank(
            q,
            g,
            metric=metric,
            k1=int(rerank.get("k1", 20)),
            k2=int(rerank.get("k2", 6)),
            lambda_value=float(rerank.get("lambda", 0.3)),
        )
    return search_distance(q, g, metric=metric, search_cfg=cfg.get("search", {}))
