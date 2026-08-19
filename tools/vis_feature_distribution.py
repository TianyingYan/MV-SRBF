#!/usr/bin/env python3
"""Create PDF/CDF plots from MV-SRBF fused or modality features."""
from __future__ import annotations

import argparse
import csv
import os
import sys

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
    parser.add_argument("--stat", type=str, default="norm", choices=["norm", "dim"])
    parser.add_argument("--dim", type=int, default=0, help="Feature dimension used when --stat dim")
    parser.add_argument("--plot", type=str, default="both", choices=["pdf", "cdf", "both"])
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--ids_per_plot", type=int, default=0, help="Use the first N sorted unique identity ids; 0 keeps all extracted ids")
    parser.add_argument("--samples_per_id", type=int, default=0, help="Samples per selected id; repeats only when an id has too few samples")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    import numpy as np
    import torch

    from data.dataloaders import build_dataloaders
    from evaluation.feature_extraction import extract_fused_feature_for_availability
    from evaluation.postprocess import maybe_normalize_features
    from tools.vis_tsne import (
        _aux_rows,
        _load_model,
        _resolve_availability,
        _resolve_modalities,
        _select_loaders,
    )
    from utils.config import apply_parameter_overrides, get_config
    from utils.reproducibility import set_seed
    from visualization.distribution import feature_distribution_values, plot_pdf_cdf
    from visualization.tsne import select_indices_by_id

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

    if not features:
        raise ValueError("No features were extracted.")
    selected_indices = select_indices_by_id(
        pids,
        ids_per_plot=args.ids_per_plot,
        samples_per_id=args.samples_per_id,
        seed=int(cfg.get("seed", 42)),
    )
    if args.max_samples > 0:
        selected_indices = selected_indices[: args.max_samples]
    if not selected_indices:
        raise ValueError("No features were selected for distribution plotting.")

    features_np = np.asarray([features[i] for i in selected_indices], dtype=np.float32)
    pids = [pids[i] for i in selected_indices]
    splits = [splits[i] for i in selected_indices]
    modalities = [modalities[i] for i in selected_indices]
    aux_rows = [aux_rows[i] for i in selected_indices]
    values = feature_distribution_values(features_np, stat=args.stat, dim=args.dim)

    os.makedirs(args.out_dir, exist_ok=True)
    stat_tag = args.stat if args.stat == "norm" else f"dim{args.dim}"
    tag = f"{args.split}_{args.feature}_{stat_tag}_{args.plot}"
    png_path = os.path.join(args.out_dir, f"feature_distribution_{tag}.png")
    csv_path = os.path.join(args.out_dir, f"feature_distribution_{tag}.csv")
    plot_pdf_cdf(
        values,
        path=png_path,
        title=f"{args.feature} feature {stat_tag} distribution",
        bins=args.bins,
        plot=args.plot,
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["value", "pid", "split", "modality", "aux"])
        for value, pid, split_name, modality, aux in zip(values, pids, splits, modalities, aux_rows):
            writer.writerow([float(value), int(pid), split_name, modality, " ".join(map(str, aux))])
    print(f"Saved feature distribution figure: {png_path}")
    print(f"Saved feature distribution CSV: {csv_path}")


if __name__ == "__main__":
    main()
