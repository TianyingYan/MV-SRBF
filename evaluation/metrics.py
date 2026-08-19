"""ReID distances, mAP@MAX, and CMC (evaluation depth controlled by MAX / max_rank)."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch


def pairwise_distance(x: torch.Tensor, y: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
    metric = metric.lower()
    if metric in {"euclidean", "l2"}:
        return torch.cdist(x, y, p=2)
    if metric not in {"cosine", "cos"}:
        raise ValueError(f"Unsupported inference metric: {metric!r}. Use cosine/cos or euclidean/l2.")
    x_n = torch.nn.functional.normalize(x, dim=1)
    y_n = torch.nn.functional.normalize(y, dim=1)
    return 1.0 - torch.mm(x_n, y_n.t())


def resolve_remove_same_aux_dims(
    remove_same_aux_dims: Sequence[int] | None = None,
) -> Tuple[int, ...]:
    """Resolve same-ID junk-filter aux dims; multiple dims use union semantics."""
    if remove_same_aux_dims is None:
        return ()
    return tuple(sorted({int(dim) for dim in remove_same_aux_dims}))


def _aux_matrix(aux: np.ndarray | None, length: int) -> np.ndarray:
    """Return a [N, A] auxiliary matrix without falling back to camera IDs."""
    if aux is None:
        return np.empty((int(length), 0), dtype=np.int64)
    arr = np.asarray(aux)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _same_aux_mask(
    q_aux_row: np.ndarray,
    g_aux: np.ndarray,
    dims: Tuple[int, ...],
) -> np.ndarray:
    if not dims:
        return np.zeros(g_aux.shape[0], dtype=bool)
    max_dim = max(dims)
    if max_dim >= q_aux_row.shape[0] or max_dim >= g_aux.shape[1]:
        raise ValueError(
            f"remove_same_aux_dims={list(dims)} exceeds available aux columns "
            f"(query={q_aux_row.shape[0]}, gallery={g_aux.shape[1]})."
        )
    same = np.zeros(g_aux.shape[0], dtype=bool)
    for dim in dims:
        same |= g_aux[:, dim] == q_aux_row[dim]
    return same


def mean_ap_at_max(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int,
    q_aux: np.ndarray | None = None,
    g_aux: np.ndarray | None = None,
    remove_same_aux_dims: Sequence[int] | None = None,
) -> float:
    """
    Mean Average Precision @ MAX (benchmark-style), following::

        mAP@MAX = (1/Q) sum_q [ 1/min(m_q, MAX) sum_{k=1}^{min(n_q, MAX)} P_q(k) rel_q(k) ]

    - ``m_q``: number of gallery images relevant to query ``q``.
    - ``n_q``: length of the ranked list after optional aux-dim junk removal.
    - ``P_q(k)``: precision at rank ``k`` = (# relevant among top-``k``) / ``k``.
    - ``rel_q(k)``: 1 if the ``k``-th prediction is relevant, else 0.

    ``remove_same_aux_dims`` drops gallery items that share the query's value on
    these aux dims **and** the same identity.

    Queries with ``m_q == 0`` are skipped (no positives in gallery).
    """
    num_q, _ = distmat.shape
    indices = np.argsort(distmat, axis=1)
    aps: list[float] = []
    dims = resolve_remove_same_aux_dims(remove_same_aux_dims)
    q_aux_m = _aux_matrix(q_aux, len(q_pids))
    g_aux_m = _aux_matrix(g_aux, len(g_pids))

    for q in range(num_q):
        q_pid = q_pids[q]
        junk_mask = (g_pids == q_pid) & _same_aux_mask(q_aux_m[q], g_aux_m, dims)
        valid_positive = (g_pids == q_pid) & ~junk_mask
        m_q = int(valid_positive.sum())
        if m_q == 0:
            continue

        order = indices[q]
        if dims:
            keep = ~junk_mask[order]
        else:
            keep = np.ones_like(order, dtype=bool)
        ranked_idx = order[keep]
        if ranked_idx.size == 0:
            continue

        is_rel = (g_pids[ranked_idx] == q_pid).astype(np.float64)
        n_q = int(is_rel.size)
        denom = float(min(m_q, max_rank))
        K = min(n_q, max_rank)

        acc = 0.0
        for k in range(1, K + 1):
            rel_k = is_rel[k - 1]
            prec_k = float(is_rel[:k].sum()) / float(k)
            acc += prec_k * rel_k

        aps.append(acc / denom)

    return float(np.mean(aps)) if aps else 0.0


def mean_ap(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = 50,
    q_aux: np.ndarray | None = None,
    g_aux: np.ndarray | None = None,
    remove_same_aux_dims: Sequence[int] | None = None,
) -> float:
    """Alias for :func:`mean_ap_at_max` (pass ``max_rank`` as MAX)."""
    return mean_ap_at_max(
        distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        max_rank,
        q_aux=q_aux,
        g_aux=g_aux,
        remove_same_aux_dims=remove_same_aux_dims,
    )


def cmc(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = 50,
    q_aux: np.ndarray | None = None,
    g_aux: np.ndarray | None = None,
    remove_same_aux_dims: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Cumulative Matching Characteristics up to ``max_rank`` (plays the role of MAX for the curve).

    After the optional aux-dim junk removal, for each query ``binary[r]=1`` iff the
    ``r``-th ranked gallery image matches ``q_pid``. Then ``cmc[r-1]`` is the mean across queries of
    ``max(binary[:r])`` (whether any correct hit appears in the top ``r`` ranks).

    ``remove_same_aux_dims`` removes same-id same-aux gallery items.
    """
    num_q, _ = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    dims = resolve_remove_same_aux_dims(remove_same_aux_dims)
    q_aux_m = _aux_matrix(q_aux, len(q_pids))
    g_aux_m = _aux_matrix(g_aux, len(g_pids))

    all_cmc = []
    for q in range(num_q):
        q_pid = q_pids[q]
        junk_mask = (g_pids == q_pid) & _same_aux_mask(q_aux_m[q], g_aux_m, dims)
        valid_positive = (g_pids == q_pid) & ~junk_mask
        if not valid_positive.any():
            continue
        order = indices[q]
        if dims:
            keep = np.invert(junk_mask[order])
        else:
            keep = np.ones_like(order, dtype=bool)
        raw_cmc = matches[q][keep]
        if raw_cmc.size == 0:
            continue
        cmc_ = raw_cmc.cumsum()
        cmc_[cmc_ > 1] = 1
        vec = cmc_[:max_rank]
        if vec.size < max_rank:
            pad = np.zeros(max_rank, dtype=vec.dtype)
            pad[: vec.size] = vec
            if vec.size > 0:
                pad[vec.size :] = vec[-1]
            vec = pad
        all_cmc.append(vec)
    if not all_cmc:
        return np.zeros(max_rank)
    return np.stack(all_cmc, axis=0).mean(axis=0)


def symmetric_metric(
    q_feat: torch.Tensor,
    g_feat: torch.Tensor,
    q_pids: torch.Tensor,
    g_pids: torch.Tensor,
    q_camids: torch.Tensor,
    g_camids: torch.Tensor,
    metric: str = "cosine",
    max_rank: int = 50,
    q_aux: torch.Tensor | None = None,
    g_aux: torch.Tensor | None = None,
    remove_same_aux_dims: Sequence[int] | None = None,
) -> Tuple[float, np.ndarray]:
    """Returns symmetric mAP/CMC, computed as 0.5 * (query-to-gallery + gallery-to-query)."""
    d1 = pairwise_distance(q_feat, g_feat, metric).cpu().numpy()
    d2 = pairwise_distance(g_feat, q_feat, metric).cpu().numpy()
    qp, gp = q_pids.cpu().numpy(), g_pids.cpu().numpy()
    qc, gc = q_camids.cpu().numpy(), g_camids.cpu().numpy()
    qa = q_aux.cpu().numpy() if q_aux is not None else None
    ga = g_aux.cpu().numpy() if g_aux is not None else None

    m1 = mean_ap_at_max(
        d1,
        qp,
        gp,
        qc,
        gc,
        max_rank,
        q_aux=qa,
        g_aux=ga,
        remove_same_aux_dims=remove_same_aux_dims,
    )
    m2 = mean_ap_at_max(
        d2,
        gp,
        qp,
        gc,
        qc,
        max_rank,
        q_aux=ga,
        g_aux=qa,
        remove_same_aux_dims=remove_same_aux_dims,
    )
    c1 = cmc(
        d1,
        qp,
        gp,
        qc,
        gc,
        max_rank,
        q_aux=qa,
        g_aux=ga,
        remove_same_aux_dims=remove_same_aux_dims,
    )
    c2 = cmc(
        d2,
        gp,
        qp,
        gc,
        qc,
        max_rank,
        q_aux=ga,
        g_aux=qa,
        remove_same_aux_dims=remove_same_aux_dims,
    )
    return 0.5 * (m1 + m2), 0.5 * (c1 + c2)
