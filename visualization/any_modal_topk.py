"""Top-10 retrieval visualization for any-modal vs any-modal settings."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _chw_to_hwc_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert a float CHW tensor in [0,1] to uint8 HWC for matplotlib."""
    tensor = x.detach().cpu().float().clamp(0, 1)
    arr = tensor.numpy().transpose(1, 2, 0)
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def _denormalize_chw(x: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, device=x.device, dtype=x.dtype).view(3, 1, 1)
    return (x * std_t + mean_t).clamp(0, 1)


def modality_preview_tensors(
    modalities: Dict[str, torch.Tensor],
    modal_keys: Sequence[str],
    available_indices: Sequence[int],
    *,
    denormalize: bool = False,
    normalize_params: Optional[Dict[str, Dict[str, Sequence[float]]]] = None,
) -> List[Optional[torch.Tensor]]:
    """Return one preview tensor per modality; missing modalities are returned as None."""
    available = set(int(i) for i in available_indices)
    previews: List[Optional[torch.Tensor]] = []
    for idx, key in enumerate(modal_keys):
        if idx not in available or key not in modalities:
            previews.append(None)
            continue
        tensor = modalities[key][0]
        if denormalize:
            params = (normalize_params or {}).get(key, (normalize_params or {}).get("default", {}))
            mean = params.get("mean", [0.485, 0.456, 0.406])
            std = params.get("std", [0.229, 0.224, 0.225])
            tensor = _denormalize_chw(tensor, mean, std)
        previews.append(tensor.clamp(0, 1))
    return previews


def _draw_missing_tile(ax, label: str, hw: Tuple[int, int]) -> None:
    blank = np.full((int(hw[0]), int(hw[1]), 3), 243, dtype=np.uint8)
    ax.imshow(blank)
    ax.text(
        0.5,
        0.5,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color="#6b7280",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _set_border(ax, color: str, width: float = 2.5) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(width)
        spine.set_edgecolor(color)


def plot_multimodal_topk_grid(
    query_modal_chw: Sequence[Optional[torch.Tensor]],
    gallery_modal_chw: Sequence[Sequence[Optional[torch.Tensor]]],
    modal_names: Sequence[str],
    title: str,
    save_path: str,
    *,
    gallery_correct: Optional[Sequence[bool]] = None,
    placeholder_size: Optional[Tuple[int, int]] = None,
) -> None:
    """Save a grid where rows are modalities and columns are query plus ranked gallery samples."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # file-only rendering; never depend on an interactive GUI backend
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plot_multimodal_topk_grid") from exc

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    num_modalities = len(modal_names)
    num_gallery = min(10, len(gallery_modal_chw))
    if num_modalities == 0 or num_gallery == 0:
        return

    fig, axes = plt.subplots(
        num_modalities,
        num_gallery + 1,
        figsize=(1.55 * (num_gallery + 1), 1.75 * num_modalities),
        squeeze=False,
    )
    correctness = list(gallery_correct or [False] * num_gallery)
    inferred_size = placeholder_size
    if inferred_size is None:
        for img in list(query_modal_chw) + [x for sample in gallery_modal_chw for x in sample]:
            if img is not None:
                inferred_size = (int(img.shape[-2]), int(img.shape[-1]))
                break
    if inferred_size is None:
        inferred_size = (256, 128)

    for row, modal_name in enumerate(modal_names):
        q_ax = axes[row, 0]
        q_img = query_modal_chw[row]
        if q_img is None:
            _draw_missing_tile(q_ax, "Missing", inferred_size)
        else:
            q_ax.imshow(_chw_to_hwc_numpy(q_img))
            q_ax.set_xticks([])
            q_ax.set_yticks([])
        _set_border(q_ax, "#2563eb", width=2.0)
        q_ax.set_ylabel(modal_name, fontsize=9)
        if row == 0:
            q_ax.set_title("Query", fontsize=9)

        for col in range(num_gallery):
            ax = axes[row, col + 1]
            gal_img = gallery_modal_chw[col][row]
            if gal_img is None:
                _draw_missing_tile(ax, "Missing", inferred_size)
            else:
                ax.imshow(_chw_to_hwc_numpy(gal_img))
                ax.set_xticks([])
                ax.set_yticks([])
            _set_border(ax, "#16a34a" if correctness[col] else "#dc2626", width=2.5)
            if row == 0:
                ax.set_title(f"Rank {col + 1}", fontsize=8)

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

__all__ = [
    "modality_preview_tensors",
    "plot_multimodal_topk_grid",
]
