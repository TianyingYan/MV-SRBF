#!/usr/bin/env python3
"""Create t-SNE plots from MV-SRBF fused or modality features."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, default="", help="Optional trained checkpoint .pth")
    parser.add_argument(
        "--parameter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override or add YAML fields before visualization.",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="query_gallery", choices=["query", "gallery", "train_eval", "query_gallery", "all"])
    parser.add_argument("--feature", type=str, default="fuse", choices=["fuse", "modal"])
    parser.add_argument("--modality", type=str, default="all", help="For modal features: all, modal key, modal name, or comma-separated values")
    parser.add_argument("--available", type=str, default="full", help="For fuse features: full or comma-separated modal keys/names/0-based indices")
    parser.add_argument("--color_by", type=str, default="pid", help="pid, split, modality, or aux{index}")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--ids_per_plot", type=int, default=0, help="Use the first N sorted unique identity ids; 0 keeps all extracted ids")
    parser.add_argument("--samples_per_id", type=int, default=0, help="Samples per selected id; repeats only when an id has too few samples")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n_iter", type=int, default=1000)
    return parser.parse_args()


def _load_model(cfg, device, weights_path: str):
    import torch

    from models.mv_srbf import MVSRBF

    model = MVSRBF(cfg).to(device)
    model.eval()
    if not weights_path:
        print("t-SNE is running without a checkpoint; features use the initialized model.")
        return model
    path = os.path.abspath(weights_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"t-SNE checkpoint not found: {path}")
    print(f"Loading t-SNE checkpoint: {path}")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    print(f"Loaded t-SNE checkpoint: {path}")
    return model


def _resolve_modalities(request: str, modal_keys: List[str], modal_names: List[str]) -> List[str]:
    if request.lower() == "all":
        return modal_keys
    lookup = {k.lower(): k for k in modal_keys}
    lookup.update({name.lower(): key for key, name in zip(modal_keys, modal_names)})
    out = []
    for token in request.split(","):
        name = token.strip().lower()
        if name not in lookup:
            raise ValueError(f"Unknown modality {token!r}. Use one of {modal_keys} or {modal_names}.")
        out.append(lookup[name])
    return out


def _resolve_availability(request: str, modal_keys: List[str], modal_names: List[str]) -> Tuple[int, ...]:
    if request.lower() in {"full", "all"}:
        return tuple(range(len(modal_keys)))
    lookup = {k.lower(): idx for idx, k in enumerate(modal_keys)}
    lookup.update({name.lower(): idx for idx, name in enumerate(modal_names)})
    out = []
    for token in request.split(","):
        name = token.strip().lower()
        if name.isdigit():
            idx = int(name)
        elif name in lookup:
            idx = int(lookup[name])
        else:
            raise ValueError(f"Unknown availability token {token!r}.")
        if idx < 0 or idx >= len(modal_keys):
            raise ValueError(f"Availability index {idx} is out of range for {len(modal_keys)} modalities.")
        out.append(idx)
    return tuple(dict.fromkeys(out))


def _aux_rows(batch, idx: int) -> List[int]:
    aux_indices = batch.get("aux_indices", {})
    if not aux_indices:
        return []
    max_dim = max(int(dim) for dim in aux_indices.keys())
    return [
        int(aux_indices[dim][idx]) if dim in aux_indices else -1
        for dim in range(max_dim + 1)
    ]


def _select_loaders(loaders: Dict, split: str) -> List[Tuple[str, object]]:
    items = []
    if split in {"query", "query_gallery", "all"}:
        items.append(("query", loaders["query_loader"]))
    if split in {"gallery", "query_gallery", "all"}:
        items.append(("gallery", loaders["gallery_loader"]))
    if split in {"train_eval", "all"}:
        if "train_eval_loader" not in loaders:
            raise ValueError("train_eval split is not available because dataloader.train_eval.enabled is false.")
        items.append(("train_eval", loaders["train_eval_loader"]))
    return items


def main() -> None:
    args = _parse_args()

    import numpy as np
    import torch

    from data.dataloaders import build_dataloaders
    from evaluation.feature_extraction import extract_fused_feature_for_availability
    from evaluation.postprocess import maybe_normalize_features
    from utils.config import apply_parameter_overrides, get_config
    from utils.reproducibility import set_seed
    from visualization.tsne import compute_tsne, labels_for_color_by, plot_tsne_embedding, select_indices_by_id

    cfg = get_config(os.path.abspath(args.config))
    apply_parameter_overrides(cfg, args.parameter)
    set_seed(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_dataloaders(cfg)
    model = _load_model(cfg, device, args.weights)
    modal_keys = list(model.modal_keys)
    modal_names = [cfg["dataset"].get("modalities", {}).get(k, k) for k in modal_keys]
    selected_modalities = _resolve_modalities(args.modality, modal_keys, modal_names)
    availability = _resolve_availability(args.available, modal_keys, modal_names)
    infer_cfg = cfg.get("inference", {})
    neck_feat = infer_cfg.get("neck_feat", "before")
    feat_norm = bool(infer_cfg.get("feat_norm", True))

    features, pids, splits, modalities, aux_rows = [], [], [], [], []
    stream_limit = args.max_samples if args.ids_per_plot <= 0 and args.samples_per_id <= 0 else 0
    with torch.no_grad():
        for split_name, loader in _select_loaders(loaders, args.split):
            for batch in loader:
                batch_modalities = {k: v.to(device) for k, v in batch["modalities"].items()}
                aux_indices = {int(k): v.to(device) for k, v in batch.get("aux_indices", {}).items()}
                if args.feature == "fuse":
                    feat = extract_fused_feature_for_availability(
                        model,
                        batch_modalities,
                        modal_keys,
                        availability,
                        aux_indices=aux_indices,
                        neck_feat=neck_feat,
                        feature_stage=infer_cfg.get("feature_stage", "fusion"),
                        feature_aggregation=infer_cfg.get("feature_aggregation", "mean"),
                    )
                    feat = maybe_normalize_features(feat, feat_norm).cpu()
                    for idx in range(feat.shape[0]):
                        features.append(feat[idx].numpy())
                        pids.append(int(batch["labels"][idx]))
                        splits.append(split_name)
                        modalities.append("fuse")
                        aux_rows.append(_aux_rows(batch, idx))
                else:
                    cls_stack, _, _ = model.encode_modalities(batch_modalities, aux_indices=aux_indices)
                    cls_stack = torch.nn.functional.normalize(cls_stack, dim=-1) if feat_norm else cls_stack
                    cls_stack = cls_stack.cpu()
                    for modal_key in selected_modalities:
                        modal_idx = modal_keys.index(modal_key)
                        modal_name = modal_names[modal_idx]
                        for idx in range(cls_stack.shape[0]):
                            features.append(cls_stack[idx, modal_idx].numpy())
                            pids.append(int(batch["labels"][idx]))
                            splits.append(split_name)
                            modalities.append(modal_name)
                            aux_rows.append(_aux_rows(batch, idx))
                if stream_limit > 0 and len(features) >= stream_limit:
                    break
            if stream_limit > 0 and len(features) >= stream_limit:
                break

    if len(features) < 2:
        raise ValueError("Need at least two extracted features for t-SNE.")
    selected_indices = select_indices_by_id(
        pids,
        ids_per_plot=args.ids_per_plot,
        samples_per_id=args.samples_per_id,
        seed=int(cfg.get("seed", 42)),
    )
    if args.max_samples > 0:
        selected_indices = selected_indices[: args.max_samples]
    if len(selected_indices) < 2:
        raise ValueError("Need at least two selected features for t-SNE.")
    features_np = np.asarray([features[i] for i in selected_indices], dtype=np.float32)
    pids = [pids[i] for i in selected_indices]
    splits = [splits[i] for i in selected_indices]
    modalities = [modalities[i] for i in selected_indices]
    aux_rows = [aux_rows[i] for i in selected_indices]
    coords = compute_tsne(
        features_np,
        perplexity=args.perplexity,
        seed=int(cfg.get("seed", 42)),
        n_iter=args.n_iter,
    )
    color_labels = labels_for_color_by(
        color_by=args.color_by,
        pids=pids,
        splits=splits,
        modalities=modalities,
        aux_rows=aux_rows,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.split}_{args.feature}_{args.color_by}"
    png_path = os.path.join(args.out_dir, f"tsne_{tag}.png")
    csv_path = os.path.join(args.out_dir, f"tsne_{tag}.csv")
    plot_tsne_embedding(
        coords,
        color_labels,
        path=png_path,
        title=f"t-SNE {args.feature} features",
        legend_title=args.color_by,
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "pid", "split", "modality", "aux"])
        for xy, pid, split_name, modality, aux in zip(coords, pids, splits, modalities, aux_rows):
            writer.writerow([float(xy[0]), float(xy[1]), int(pid), split_name, modality, " ".join(map(str, aux))])
    print(f"Saved t-SNE figure: {png_path}")
    print(f"Saved t-SNE CSV: {csv_path}")


if __name__ == "__main__":
    main()
