#!/usr/bin/env python3
"""Create Top-10 retrieval figures for any-modal retrieval scenarios."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, default="", help="Optional checkpoint .pth")
    parser.add_argument(
        "--parameter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override or add YAML fields before visualization.",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_queries", type=int, default=5)
    parser.add_argument("--metric", type=str, default="", help="Override inference.metric; defaults to the YAML value")
    parser.add_argument("--gallery_cap", type=int, default=5000, help="Max gallery images to cache")
    args = parser.parse_args()

    import numpy as np
    import torch

    from data.dataloaders import build_dataloaders
    from evaluation.modality_utils import list_any_modal_rank_scenarios, scenario_display_name
    from evaluation.postprocess import inference_distance
    from evaluation.feature_extraction import extract_fused_feature_for_availability
    from evaluation.topk_rank import topk_gallery_indices
    from models.mv_srbf import MVSRBF
    from utils.config import apply_parameter_overrides, get_config
    from utils.input_config import resolve_input_size
    from utils.reproducibility import set_seed
    from visualization.any_modal_topk import modality_preview_tensors, plot_multimodal_topk_grid

    cfg = get_config(os.path.abspath(args.config))
    apply_parameter_overrides(cfg, args.parameter)
    set_seed(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", True)))
    input_size = resolve_input_size(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_dataloaders(cfg)
    q_loader = loaders["query_loader"]
    g_loader = loaders["gallery_loader"]

    model = MVSRBF(cfg).to(device)
    model.eval()
    if args.weights:
        weights_path = os.path.abspath(args.weights)
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Visualization checkpoint not found: {weights_path}")
        print(f"Loading visualization checkpoint: {weights_path}")
        try:
            ckpt = torch.load(weights_path, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(weights_path, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        print(f"Loaded visualization checkpoint: {weights_path}")
    else:
        print("Visualization is running without a checkpoint; retrieval examples use the initialized model.")

    modal_keys = list(model.modal_keys)
    modal_names = [cfg["dataset"].get("modalities", {}).get(k, k) for k in modal_keys]

    norm_cfg = cfg.get("input", {}).get("normalize", {})
    infer_cfg = cfg.get("inference", {})
    remove_same_aux_dims = infer_cfg.get("remove_same_aux_dims", None)
    metric = args.metric or infer_cfg.get("metric", "cosine")

    queries: list[dict] = []
    for batch in q_loader:
        for idx in range(batch["labels"].shape[0]):
            if len(queries) >= args.max_queries:
                break
            queries.append(
                {
                    "modalities": {k: v[idx : idx + 1] for k, v in batch["modalities"].items()},
                    "aux_indices": {
                        int(k): v[idx : idx + 1]
                        for k, v in batch.get("aux_indices", {}).items()
                    },
                    "pid": int(batch["labels"][idx]),
                    "camid": int(batch["camids"][idx]) if "camids" in batch else 0,
                }
            )
        if len(queries) >= args.max_queries:
            break

    gallery_list: list[dict] = []
    for batch in g_loader:
        for idx in range(batch["labels"].shape[0]):
            if len(gallery_list) >= args.gallery_cap:
                break
            gallery_list.append(
                {
                    "modalities": {k: v[idx : idx + 1] for k, v in batch["modalities"].items()},
                    "aux_indices": {
                        int(k): v[idx : idx + 1]
                        for k, v in batch.get("aux_indices", {}).items()
                    },
                    "pid": int(batch["labels"][idx]),
                    "camid": int(batch["camids"][idx]) if "camids" in batch else 0,
                }
            )
        if len(gallery_list) >= args.gallery_cap:
            break

    def feats_from_list(items: list[dict], avail: tuple[int, ...]) -> torch.Tensor:
        outs = []
        with torch.no_grad():
            for item in items:
                modalities = {k: v.to(device) for k, v in item["modalities"].items()}
                aux_indices = {int(k): v.to(device) for k, v in item.get("aux_indices", {}).items()}
                outs.append(
                    extract_fused_feature_for_availability(
                        model,
                        modalities,
                        modal_keys,
                        avail,
                        aux_indices=aux_indices,
                        neck_feat=infer_cfg.get("neck_feat", "before"),
                        feature_stage=infer_cfg.get("feature_stage", "fusion"),
                        feature_aggregation=infer_cfg.get("feature_aggregation", "mean"),
                    ).cpu()
                )
        return torch.cat(outs, dim=0)

    def aux_row(item: dict) -> list[int]:
        aux_indices = item.get("aux_indices", {})
        if not aux_indices:
            return []
        max_dim = max(int(dim) for dim in aux_indices.keys())
        row = []
        for dim in range(max_dim + 1):
            value = aux_indices.get(dim)
            row.append(int(value.reshape(-1)[0]) if value is not None else -1)
        return row

    q_pids_arr = np.array([item["pid"] for item in queries], dtype=np.int64)
    q_aux_arr = np.array([aux_row(item) for item in queries], dtype=np.int64)
    g_pids_arr = np.array([item["pid"] for item in gallery_list], dtype=np.int64)
    g_aux_arr = np.array([aux_row(item) for item in gallery_list], dtype=np.int64)
    os.makedirs(args.out_dir, exist_ok=True)

    for scenario in list_any_modal_rank_scenarios(len(modal_keys)):
        q_avail = scenario["q_avail"]
        g_avail = scenario["g_avail"]
        title = scenario_display_name(scenario, modal_names=modal_names)

        q_feat = feats_from_list(queries, q_avail).to(device)
        g_feat = feats_from_list(gallery_list, g_avail).to(device)
        dist = inference_distance(q_feat, g_feat, metric, infer_cfg).cpu().numpy()
        top_idx = topk_gallery_indices(
            dist,
            q_pids_arr,
            g_pids_arr,
            topk=10,
            q_aux=q_aux_arr,
            g_aux=g_aux_arr,
            remove_same_aux_dims=remove_same_aux_dims,
        )

        for qi in range(q_feat.shape[0]):
            q_vis = modality_preview_tensors(
                queries[qi]["modalities"],
                modal_keys,
                q_avail,
                denormalize=True,
                normalize_params=norm_cfg,
            )
            gal_chw = []
            correctness: list[bool] = []
            for rank in range(10):
                gj = int(top_idx[qi, rank])
                if gj < 0 or gj >= len(gallery_list):
                    break
                gal_chw.append(
                    modality_preview_tensors(
                        gallery_list[gj]["modalities"],
                        modal_keys,
                        g_avail,
                        denormalize=True,
                        normalize_params=norm_cfg,
                    )
                )
                correctness.append(gallery_list[gj]["pid"] == queries[qi]["pid"])

            slug = f'{scenario["tag"]}_q{"".join(map(str, q_avail))}_g{"".join(map(str, g_avail))}_q{qi}'
            path = os.path.join(args.out_dir, f"{slug}.png")
            plot_multimodal_topk_grid(
                q_vis,
                gal_chw,
                modal_names,
                f"{title} | query #{qi}",
                path,
                gallery_correct=correctness,
                placeholder_size=input_size,
            )


if __name__ == "__main__":
    main()
