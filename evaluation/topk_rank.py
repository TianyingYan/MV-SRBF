"""Top-K gallery indices for retrieval visualization."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from evaluation.metrics import _aux_matrix, _same_aux_mask, resolve_remove_same_aux_dims


def topk_gallery_indices(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    topk: int = 10,
    q_aux: np.ndarray | None = None,
    g_aux: np.ndarray | None = None,
    remove_same_aux_dims: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Per query: ascending distance, optionally skip same-id gallery entries with matching aux dims,
    return indices of first ``topk`` valid gallery images (-1 if fewer than topk valid).

    ``remove_same_aux_dims`` skips same-id same-aux gallery entries.
    """
    num_q, num_g = distmat.shape
    out = np.full((num_q, topk), -1, dtype=np.int64)
    indices = np.argsort(distmat, axis=1)
    dims = resolve_remove_same_aux_dims(remove_same_aux_dims)
    q_aux_m = _aux_matrix(q_aux, len(q_pids))
    g_aux_m = _aux_matrix(g_aux, len(g_pids))
    for q in range(num_q):
        q_pid = q_pids[q]
        same_aux = _same_aux_mask(q_aux_m[q], g_aux_m, dims)
        taken = 0
        for j in indices[q]:
            if dims and g_pids[j] == q_pid and same_aux[j]:
                continue
            out[q, taken] = j
            taken += 1
            if taken >= topk:
                break
    return out
