"""t-SNE plotting helpers for ReID feature embeddings."""
from __future__ import annotations

import inspect
from typing import Iterable, Sequence

import numpy as np


def select_indices_by_id(
    pids: Sequence[int],
    *,
    ids_per_plot: int = 0,
    samples_per_id: int = 0,
    seed: int = 42,
) -> list[int]:
    """
    Select feature rows by sorted unique identity ids.

    ``samples_per_id`` keeps natural sample order and duplicates only when an
    identity has fewer rows than requested.
    """
    ids = sorted({int(pid) for pid in pids})
    if ids_per_plot > 0:
        ids = ids[: int(ids_per_plot)]
    by_id = {pid: [] for pid in ids}
    for idx, pid in enumerate(pids):
        pid_int = int(pid)
        if pid_int in by_id:
            by_id[pid_int].append(idx)

    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    for pid in ids:
        idxs = list(by_id[pid])
        if samples_per_id > 0:
            target = int(samples_per_id)
            if len(idxs) >= target:
                idxs = idxs[:target]
            elif idxs:
                extra = rng.choice(idxs, size=target - len(idxs), replace=True).astype(int).tolist()
                idxs.extend(extra)
        selected.extend(idxs)
    return selected


def compute_tsne(
    features: np.ndarray,
    *,
    perplexity: float = 30.0,
    seed: int = 42,
    n_iter: int = 1000,
) -> np.ndarray:
    """Run sklearn t-SNE and return two-dimensional coordinates."""
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError(
            "t-SNE visualization requires scikit-learn. Install it with `pip install scikit-learn`."
        ) from exc

    if features.ndim != 2:
        raise ValueError(f"features must be a 2D array, got shape {features.shape}.")
    if features.shape[0] < 2:
        raise ValueError("t-SNE requires at least two samples.")
    safe_perplexity = min(float(perplexity), max(1.0, (features.shape[0] - 1) / 3.0))
    kwargs = {
        "n_components": 2,
        "perplexity": safe_perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": int(seed),
    }
    iter_name = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    kwargs[iter_name] = int(n_iter)
    tsne = TSNE(**kwargs)
    return tsne.fit_transform(features)


def plot_tsne_embedding(
    coords: np.ndarray,
    labels: Sequence,
    *,
    path: str,
    title: str = "t-SNE",
    legend_title: str = "label",
    max_legend_items: int = 30,
) -> None:
    """Plot two-dimensional t-SNE coordinates with categorical labels."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # file-only rendering; never depend on an interactive GUI backend
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for t-SNE plotting.") from exc

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape [N, 2], got {coords.shape}.")
    labels_arr = np.asarray(list(labels))
    if coords.shape[0] != labels_arr.shape[0]:
        raise ValueError("coords and labels must contain the same number of samples.")

    fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
    unique = list(dict.fromkeys(labels_arr.tolist()))
    cmap = plt.get_cmap("tab20", max(1, min(20, len(unique))))
    for idx, label in enumerate(unique):
        mask = labels_arr == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.82,
            linewidths=0.0,
            color=cmap(idx % 20),
            label=str(label),
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if len(unique) <= max_legend_items:
        ax.legend(title=legend_title, fontsize=7, title_fontsize=8, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def labels_for_color_by(
    *,
    color_by: str,
    pids: Sequence[int],
    splits: Sequence[str],
    modalities: Sequence[str],
    aux_rows: Iterable[Sequence[int]],
) -> list:
    """Resolve plot labels from collected feature metadata."""
    if color_by == "pid":
        return [int(x) for x in pids]
    if color_by == "split":
        return [str(x) for x in splits]
    if color_by == "modality":
        return [str(x) for x in modalities]
    if color_by.startswith("aux"):
        dim = int(color_by[3:])
        labels = []
        for row in aux_rows:
            labels.append(int(row[dim]) if dim < len(row) else -1)
        return labels
    raise ValueError("color_by must be pid, split, modality, or aux{index}.")
