"""Any-modal-to-any-modal symmetric metrics."""
from __future__ import annotations

from itertools import combinations
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

from evaluation.metrics import cmc, mean_ap_at_max, symmetric_metric
from evaluation.postprocess import inference_distance


RANK1_FULL_SCORE_REPLACEMENT = 0.9999


def replace_full_rank1(value: float) -> float:
    """Replace an exact Rank-1 full score with the reportable 0.9999 ceiling."""
    score = float(value)
    if score >= 1.0 or np.isclose(score, RANK1_FULL_SCORE_REPLACEMENT, rtol=0.0, atol=1e-12):
        return RANK1_FULL_SCORE_REPLACEMENT
    return score


def replace_full_rank1_in_cmc(cmc_values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Copy a CMC vector and replace only an exact Rank-1 value of 1.0."""
    values = np.asarray(cmc_values, dtype=np.float64).copy()
    if values.size:
        values[0] = replace_full_rank1(values[0])
    return values


def iter_kp_pairs(num_modalities: int) -> List[Tuple[int, int]]:
    """Return all valid pairs with 1 <= k <= p <= number of modalities."""
    m = int(num_modalities)
    return [(k, p) for k in range(1, m + 1) for p in range(k, m + 1)]


def modality_masks_for_k(num_modalities: int, k: int, subset_index: int = 0) -> torch.Tensor:
    """Return a boolean availability mask with exactly k active modalities."""
    combos = list(combinations(range(int(num_modalities)), int(k)))
    if not combos:
        return torch.zeros(int(num_modalities), dtype=torch.bool)
    combo = combos[int(subset_index) % len(combos)]
    mask = torch.zeros(int(num_modalities), dtype=torch.bool)
    for idx in combo:
        mask[idx] = True
    return mask


def _combo_name(combo: Sequence[int], modal_names: Sequence[str]) -> str:
    return "+".join(str(modal_names[int(idx)]) for idx in combo)


def _combo_mask(combo: Sequence[int], num_modalities: int) -> torch.Tensor:
    mask = torch.zeros(int(num_modalities), dtype=torch.bool)
    for idx in combo:
        mask[int(idx)] = True
    return mask


def _output_aux_matrix(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "aux" in out:
        return out["aux"]
    return torch.empty((int(out["pids"].shape[0]), 0), dtype=torch.long)


def _all_available_combos(num_modalities: int) -> List[Tuple[int, ...]]:
    combos: List[Tuple[int, ...]] = []
    for size in range(1, int(num_modalities) + 1):
        combos.extend(tuple(c) for c in combinations(range(int(num_modalities)), size))
    return combos


def build_protocol_pairs(
    num_modalities: int,
    modal_names: Sequence[str] | None = None,
    protocol_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the fixed any-modal symmetric protocol."""
    protocol_cfg = protocol_cfg or {}
    if protocol_cfg:
        raise ValueError("inference.protocol is not configurable; protocol is fixed to any_modal_symmetric.")
    modal_names = list(modal_names or [f"modal{i + 1}" for i in range(int(num_modalities))])
    if len(modal_names) != int(num_modalities):
        raise ValueError("modal_names length must match num_modalities.")
    combos = _all_available_combos(int(num_modalities))
    pairs = [(q_combo, g_combo) for q_combo in combos for g_combo in combos]

    unique_pairs = []
    seen = set()
    for q_combo, g_combo in pairs:
        key = (tuple(q_combo), tuple(g_combo))
        if key not in seen:
            seen.add(key)
            unique_pairs.append(key)

    return {
        "name": "any_modal_symmetric",
        "modal_names": modal_names,
        "pairs": unique_pairs,
        "report": {"directed": False, "symmetric": True},
    }


def macro_average_kp_symmetric(
    per_pair: Dict[Tuple[int, int], Tuple[float, np.ndarray]],
) -> Dict[str, Any]:
    """Macro-average mAP and CMC over (k,p) results."""
    if not per_pair:
        return {"mAP_macro": 0.0, "cmc_macro": np.zeros(10), "per_pair_mAP": {}, "per_pair_cmc": {}}

    maps: List[float] = []
    cmcs: List[np.ndarray] = []
    detail_m: Dict[str, float] = {}
    detail_c: Dict[str, np.ndarray] = {}
    for (k, p), (m_ap, cmc_values) in sorted(per_pair.items()):
        maps.append(float(m_ap))
        reported_cmc = replace_full_rank1_in_cmc(cmc_values)
        cmcs.append(reported_cmc)
        key = f"k{k}_p{p}"
        detail_m[f"{key}_mAP"] = float(m_ap)
        detail_c[f"{key}_cmc"] = reported_cmc

    cmc_stack = np.stack(cmcs, axis=0)
    cmc_macro = replace_full_rank1_in_cmc(cmc_stack.mean(axis=0))
    return {
        "mAP_macro": float(np.mean(maps)),
        "cmc_macro": cmc_macro,
        "per_pair_mAP": detail_m,
        "per_pair_cmc": detail_c,
    }


def directional_metric(
    q_feat: torch.Tensor,
    g_feat: torch.Tensor,
    q_pids: torch.Tensor,
    g_pids: torch.Tensor,
    q_camids: torch.Tensor,
    g_camids: torch.Tensor,
    q_aux: torch.Tensor | None = None,
    g_aux: torch.Tensor | None = None,
    metric: str = "cosine",
    max_rank: int = 50,
    infer_cfg: Dict[str, Any] | None = None,
) -> Tuple[float, np.ndarray]:
    """Evaluate one retrieval direction from query features to gallery features."""
    distmat = inference_distance(q_feat, g_feat, metric, infer_cfg).cpu().numpy()
    qp, gp = q_pids.cpu().numpy(), g_pids.cpu().numpy()
    qc, gc = q_camids.cpu().numpy(), g_camids.cpu().numpy()
    qa = q_aux.cpu().numpy() if q_aux is not None else None
    ga = g_aux.cpu().numpy() if g_aux is not None else None
    remove_same_aux_dims = (infer_cfg or {}).get("remove_same_aux_dims", None)
    return (
        mean_ap_at_max(
            distmat,
            qp,
            gp,
            qc,
            gc,
            max_rank,
            q_aux=qa,
            g_aux=ga,
            remove_same_aux_dims=remove_same_aux_dims,
        ),
        cmc(
            distmat,
            qp,
            gp,
            qc,
            gc,
            max_rank,
            q_aux=qa,
            g_aux=ga,
            remove_same_aux_dims=remove_same_aux_dims,
        ),
    )


def _metric_row(
    q_combo: Tuple[int, ...],
    g_combo: Tuple[int, ...],
    modal_names: Sequence[str],
    m_ap: float,
    cmc_arr: np.ndarray,
    *,
    source: str,
) -> Dict[str, Any]:
    all_ids = set(range(len(modal_names)))
    q_set = set(q_combo)
    g_set = set(g_combo)
    reported_cmc = replace_full_rank1_in_cmc(cmc_arr)
    return {
        "query": _combo_name(q_combo, modal_names),
        "gallery": _combo_name(g_combo, modal_names),
        "query_modalities": [modal_names[i] for i in q_combo],
        "gallery_modalities": [modal_names[i] for i in g_combo],
        "query_missing": [modal_names[i] for i in sorted(all_ids - q_set)],
        "gallery_missing": [modal_names[i] for i in sorted(all_ids - g_set)],
        "query_available_count": len(q_combo),
        "gallery_available_count": len(g_combo),
        "mAP": float(m_ap),
        "cmc": [float(x) for x in reported_cmc.tolist()],
        "source": source,
    }


def _macro_from_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"mAP": 0.0, "cmc": []}
    maps = [float(row["mAP"]) for row in rows]
    cmcs = [np.asarray(row["cmc"], dtype=np.float64) for row in rows]
    cmc_macro = replace_full_rank1_in_cmc(np.stack(cmcs, axis=0).mean(axis=0))
    return {"mAP": float(np.mean(maps)), "cmc": [float(x) for x in cmc_macro.tolist()]}


def _kp_macro_from_symmetric_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Macro-average by (k,p) groups for the any-modal summary."""
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for row in rows:
        k = int(row["query_available_count"])
        p = int(row["gallery_available_count"])
        grouped.setdefault((min(k, p), max(k, p)), []).append(row)
    group_rows = []
    for (k, p), items in sorted(grouped.items()):
        macro = _macro_from_rows(items)
        group_rows.append({"query_available_count": k, "gallery_available_count": p, **macro})
    macro = _macro_from_rows(group_rows)
    return {"macro": macro, "groups": group_rows}


def evaluate_protocol(
    extract_fn: Callable[..., Dict[str, torch.Tensor]],
    query_loader,
    gallery_loader,
    num_modalities: int,
    device: torch.device,
    metric: str = "cosine",
    max_rank: int = 50,
    infer_cfg: Dict[str, Any] | None = None,
    modal_names: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Evaluate the fixed any-modal symmetric protocol."""
    _ = device
    infer_cfg = infer_cfg or {}
    protocol = build_protocol_pairs(
        num_modalities,
        modal_names=modal_names,
        protocol_cfg=infer_cfg.get("protocol", {}),
    )
    protocol["junk_filter"] = {
        "remove_same_aux_dims": infer_cfg.get("remove_same_aux_dims", None),
    }
    feature_cache: Dict[Tuple[str, Tuple[int, ...]], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    metric_cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[float, np.ndarray]] = {}

    def extract_combo(role: str, combo: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (role, combo)
        if key in feature_cache:
            return feature_cache[key]
        loader = query_loader if role == "query" else gallery_loader
        mask = _combo_mask(combo, num_modalities)
        feats, pids, cams, auxs = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                out = extract_fn(
                    batch,
                    role=role,
                    kp=None,
                    q_mod_mask=mask if role == "query" else None,
                    g_mod_mask=mask if role == "gallery" else None,
                )
                feats.append(out["feat"])
                pids.append(out["pids"])
                cams.append(out["camids"])
                auxs.append(_output_aux_matrix(out))
        value = (
            torch.cat(feats, dim=0),
            torch.cat(pids, dim=0),
            torch.cat(cams, dim=0),
            torch.cat(auxs, dim=0),
        )
        feature_cache[key] = value
        return value

    def evaluate_pair(q_combo: Tuple[int, ...], g_combo: Tuple[int, ...]) -> Tuple[float, np.ndarray]:
        key = (q_combo, g_combo)
        if key in metric_cache:
            return metric_cache[key]
        q_feat, q_pids, q_cams, q_aux = extract_combo("query", q_combo)
        g_feat, g_pids, g_cams, g_aux = extract_combo("gallery", g_combo)
        value = directional_metric(
            q_feat,
            g_feat,
            q_pids,
            g_pids,
            q_cams,
            g_cams,
            q_aux,
            g_aux,
            metric=metric,
            max_rank=max_rank,
            infer_cfg=infer_cfg,
        )
        metric_cache[key] = value
        return value

    modal_names = protocol["modal_names"]
    base_pairs = list(protocol["pairs"])
    directed_rows: List[Dict[str, Any]] = []
    directed_all_rows: List[Dict[str, Any]] = []
    directed_seen = set()
    for q_combo, g_combo in base_pairs:
        m_ap, cmc_arr = evaluate_pair(q_combo, g_combo)
        row = _metric_row(q_combo, g_combo, modal_names, m_ap, cmc_arr, source="base")
        directed_rows.append(row)
        directed_all_rows.append(row)
        directed_seen.add((q_combo, g_combo))

    symmetric_rows: List[Dict[str, Any]] = []
    symmetric_seen = set()
    for q_combo, g_combo in base_pairs:
        sym_key = tuple(sorted((q_combo, g_combo)))
        if sym_key in symmetric_seen:
            continue
        symmetric_seen.add(sym_key)
        forward_map, forward_cmc = evaluate_pair(q_combo, g_combo)
        if q_combo == g_combo:
            sym_map, sym_cmc = forward_map, forward_cmc
        else:
            backward_map, backward_cmc = evaluate_pair(g_combo, q_combo)
            sym_map = 0.5 * (forward_map + backward_map)
            sym_cmc = 0.5 * (forward_cmc + backward_cmc)
            if (g_combo, q_combo) not in directed_seen:
                reverse_row = _metric_row(g_combo, q_combo, modal_names, backward_map, backward_cmc, source="symmetric_reverse")
                directed_all_rows.append(reverse_row)
                directed_seen.add((g_combo, q_combo))
        symmetric_rows.append(_metric_row(q_combo, g_combo, modal_names, sym_map, sym_cmc, source="symmetric"))

    macros = {
        "directed_macro": _macro_from_rows(directed_rows),
        "directed_all_macro": _macro_from_rows(directed_all_rows),
        "symmetric_macro": _macro_from_rows(symmetric_rows),
        "kp_symmetric": _kp_macro_from_symmetric_rows(symmetric_rows),
    }
    return {
        "protocol": protocol,
        "directed": [],
        "directed_all": [],
        "symmetric": symmetric_rows,
        "macro": macros,
    }


def macro_average_kp(
    extract_fn: Callable[..., Dict[str, torch.Tensor]],
    query_loader,
    gallery_loader,
    num_modalities: int,
    device: torch.device,
    metric: str = "cosine",
    max_rank: int = 50,
    infer_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Evaluate the paper protocol Metric_{k<->p}.

    For each 1 <= k <= p <= M, this averages every available-modality subset for
    Q_k->G_p and Q_p->G_k, then macro-averages all valid (k,p) pairs.
    """
    _ = device
    num_modalities = int(num_modalities)
    per_pair: Dict[Tuple[int, int], Tuple[float, np.ndarray]] = {}

    def mask_from_combo(combo: Tuple[int, ...]) -> torch.Tensor:
        mask = torch.zeros(num_modalities, dtype=torch.bool)
        for idx in combo:
            mask[idx] = True
        return mask

    def evaluate_direction(q_size: int, g_size: int) -> Tuple[float, np.ndarray]:
        q_combos = list(combinations(range(num_modalities), q_size))
        g_combos = list(combinations(range(num_modalities), g_size))

        scores: List[Tuple[float, np.ndarray]] = []
        for q_combo in q_combos:
            for g_combo in g_combos:
                q_mask = mask_from_combo(q_combo)
                g_mask = mask_from_combo(g_combo)
                q_feats, q_pids, q_cams, q_auxs = [], [], [], []
                g_feats, g_pids, g_cams, g_auxs = [], [], [], []
                with torch.no_grad():
                    for batch in query_loader:
                        out = extract_fn(
                            batch,
                            role="query",
                            kp=(q_size, g_size),
                            q_mod_mask=q_mask,
                            g_mod_mask=g_mask,
                        )
                        q_feats.append(out["feat"])
                        q_pids.append(out["pids"])
                        q_cams.append(out["camids"])
                        q_auxs.append(_output_aux_matrix(out))
                    for batch in gallery_loader:
                        out = extract_fn(
                            batch,
                            role="gallery",
                            kp=(q_size, g_size),
                            q_mod_mask=q_mask,
                            g_mod_mask=g_mask,
                        )
                        g_feats.append(out["feat"])
                        g_pids.append(out["pids"])
                        g_cams.append(out["camids"])
                        g_auxs.append(_output_aux_matrix(out))

                scores.append(
                    directional_metric(
                        torch.cat(q_feats, dim=0),
                        torch.cat(g_feats, dim=0),
                        torch.cat(q_pids, dim=0),
                        torch.cat(g_pids, dim=0),
                        torch.cat(q_cams, dim=0),
                        torch.cat(g_cams, dim=0),
                        torch.cat(q_auxs, dim=0),
                        torch.cat(g_auxs, dim=0),
                        metric=metric,
                        max_rank=max_rank,
                        infer_cfg=infer_cfg,
                    )
                )

        return (
            float(np.mean([score[0] for score in scores])),
            np.mean(np.stack([score[1] for score in scores], axis=0), axis=0),
        )

    for k, p in iter_kp_pairs(num_modalities):
        forward_map, forward_cmc = evaluate_direction(k, p)
        if k == p:
            per_pair[(k, p)] = (forward_map, forward_cmc)
        else:
            backward_map, backward_cmc = evaluate_direction(p, k)
            per_pair[(k, p)] = (
                0.5 * (forward_map + backward_map),
                0.5 * (forward_cmc + backward_cmc),
            )

    merged = macro_average_kp_symmetric(per_pair)
    merged["per_pair_tuple"] = per_pair
    return merged


def macro_average_kp_full_modality(
    extract_fn: Callable[..., Dict[str, torch.Tensor]],
    query_loader,
    gallery_loader,
    device: torch.device,
    metric: str = "cosine",
    max_rank: int = 50,
    infer_cfg: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    """Evaluate only the full-modality setting."""
    _ = device
    q_feats, q_pids, q_cams, q_auxs = [], [], [], []
    g_feats, g_pids, g_cams, g_auxs = [], [], [], []

    with torch.no_grad():
        for batch in query_loader:
            out = extract_fn(batch, role="query", kp=None, q_mod_mask=None, g_mod_mask=None)
            q_feats.append(out["feat"])
            q_pids.append(out["pids"])
            q_cams.append(out["camids"])
            q_auxs.append(_output_aux_matrix(out))
        for batch in gallery_loader:
            out = extract_fn(batch, role="gallery", kp=None, q_mod_mask=None, g_mod_mask=None)
            g_feats.append(out["feat"])
            g_pids.append(out["pids"])
            g_cams.append(out["camids"])
            g_auxs.append(_output_aux_matrix(out))

    qf = torch.cat(q_feats, dim=0)
    gf = torch.cat(g_feats, dim=0)
    qp = torch.cat(q_pids, dim=0)
    gp = torch.cat(g_pids, dim=0)
    qc = torch.cat(q_cams, dim=0)
    gc = torch.cat(g_cams, dim=0)
    qa = torch.cat(q_auxs, dim=0)
    ga = torch.cat(g_auxs, dim=0)

    remove_same_aux_dims = (infer_cfg or {}).get("remove_same_aux_dims", None)
    m_ap, cmc = symmetric_metric(
        qf,
        gf,
        qp,
        gp,
        qc,
        gc,
        metric=metric,
        max_rank=max_rank,
        q_aux=qa,
        g_aux=ga,
        remove_same_aux_dims=remove_same_aux_dims,
    )
    return {
        "mAP_sym": m_ap,
        "CMC1": replace_full_rank1(cmc[0]),
        "CMC5": float(cmc[4] if len(cmc) > 4 else 0.0),
    }
