"""PDF/CDF plotting helpers for ReID feature distributions."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def feature_distribution_values(features: np.ndarray, *, stat: str = "norm", dim: int = 0) -> np.ndarray:
    """Reduce feature vectors to scalar values for distribution plots."""
    if features.ndim != 2:
        raise ValueError(f"features must have shape [N, D], got {features.shape}.")
    stat = str(stat).lower()
    if stat == "norm":
        return np.linalg.norm(features, axis=1)
    if stat == "dim":
        dim = int(dim)
        if dim < 0 or dim >= features.shape[1]:
            raise ValueError(f"Feature dim {dim} is out of range for D={features.shape[1]}.")
        return features[:, dim]
    raise ValueError("stat must be 'norm' or 'dim'.")


def plot_pdf_cdf(
    values: Sequence[float],
    *,
    path: str,
    title: str = "Feature Distribution",
    bins: int = 50,
    plot: str = "both",
) -> None:
    """Plot empirical PDF and/or CDF from scalar feature values."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # file-only rendering; never depend on an interactive GUI backend
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for feature distribution plotting.") from exc

    values_np = np.asarray(list(values), dtype=np.float64)
    if values_np.size == 0:
        raise ValueError("Need at least one value for a distribution plot.")
    if not np.isfinite(values_np).all():
        raise ValueError("Feature distribution values contain NaN or Inf.")

    plot = str(plot).lower()
    if plot not in {"pdf", "cdf", "both"}:
        raise ValueError("plot must be 'pdf', 'cdf', or 'both'.")

    if plot == "both":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=180)
    else:
        fig, ax = plt.subplots(figsize=(5, 4), dpi=180)
        axes = [ax]

    axis_idx = 0
    if plot in {"pdf", "both"}:
        ax = axes[axis_idx]
        axis_idx += 1
        ax.hist(values_np, bins=int(bins), density=True, color="#376092", alpha=0.82)
        ax.set_title("PDF")
        ax.set_xlabel("value")
        ax.set_ylabel("density")

    if plot in {"cdf", "both"}:
        ax = axes[axis_idx]
        sorted_values = np.sort(values_np)
        probs = np.arange(1, sorted_values.size + 1, dtype=np.float64) / sorted_values.size
        ax.plot(sorted_values, probs, color="#9b2c2c", linewidth=2.0)
        ax.set_title("CDF")
        ax.set_xlabel("value")
        ax.set_ylabel("probability")
        ax.set_ylim(0.0, 1.02)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
