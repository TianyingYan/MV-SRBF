"""Build MV-SRBF masks from available modality indices."""
from __future__ import annotations

from itertools import combinations
from typing import List, Sequence, Tuple

import torch


def availability_to_cls_mask(
    batch_size: int,
    num_modalities: int,
    available_indices: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """
    Return a model mask with True where the modality token is missing.

    available_indices contains modality indices in 0..M-1 that are present.
    """
    mask = torch.ones(batch_size, num_modalities, dtype=torch.bool, device=device)
    for idx in available_indices:
        if 0 <= idx < num_modalities:
            mask[:, idx] = False
    return mask


def subsets_of_size(num_modalities: int, k: int) -> List[Tuple[int, ...]]:
    """Return all k-subsets as sorted tuples."""
    return list(combinations(range(num_modalities), k))


def list_any_modal_rank_scenarios(num_modalities: int) -> List[dict]:
    """Return every directed query/gallery availability scenario for any modality count."""
    entries: List[dict] = []
    total = int(num_modalities)
    for q_size in range(1, total + 1):
        for g_size in range(1, total + 1):
            tag = f"{q_size}v{g_size}"
            for q_avail in subsets_of_size(total, q_size):
                for g_avail in subsets_of_size(total, g_size):
                    entries.append(
                        {
                            "tag": tag,
                            "label": tag,
                            "q_avail": tuple(q_avail),
                            "g_avail": tuple(g_avail),
                        }
                    )
    return entries


def list_three_modality_rank_scenarios() -> List[dict]:
    """Return every directed query/gallery availability setting for three modalities."""
    return list_any_modal_rank_scenarios(3)


def scenario_display_name(entry: dict, modal_names: Sequence[str] | None = None) -> str:
    """Return a readable title for an asymmetric modality setting."""

    def fmt(ids: Tuple[int, ...]) -> str:
        if modal_names and len(modal_names) > max(ids, default=-1):
            return "+".join(modal_names[i] for i in ids)
        return ",".join(str(i + 1) for i in ids)

    return f'{entry["tag"]} | Q[{fmt(tuple(entry["q_avail"]))}] -> G[{fmt(tuple(entry["g_avail"]))}]'
