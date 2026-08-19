import torch
from typing import Any, Dict, List, Tuple, Union


def collate_multispectral_batch(
    batch: List[Tuple[Any, ...]],
) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    Collate MultiSpectralReIDImageDataset items: each __getitem__ returns
    (modalities_dict, oid, *extras) where modalities_dict maps keys to tensors [C,H,W].
    """
    if not batch:
        raise ValueError("empty batch")

    modal0 = batch[0][0]
    if not isinstance(modal0, dict):
        raise TypeError(f"Expected modalities dict, got {type(modal0)}")

    keys = sorted(modal0.keys())
    modalities: Dict[str, torch.Tensor] = {}
    for k in keys:
        modalities[k] = torch.stack([item[0][k] for item in batch], dim=0)

    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    out: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]] = {
        "modalities": modalities,
        "labels": labels,
    }
    if len(batch[0]) > 2:
        aux_indices = {}
        for aux_dim in range(len(batch[0]) - 2):
            aux_indices[aux_dim] = torch.tensor([item[2 + aux_dim] for item in batch], dtype=torch.long)
        out["aux_indices"] = aux_indices
        out["camids"] = aux_indices[0]
    return out

