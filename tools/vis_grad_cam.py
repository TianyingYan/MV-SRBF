#!/usr/bin/env python3
"""Create Grad-CAM overlays for encoder or fusion classification stages."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Iterable, List

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
    parser.add_argument("--split", type=str, default="query", choices=["query", "gallery", "train_eval"])
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--modality", type=str, default="all", help="all, a modal key, a modal name, or comma-separated values")
    parser.add_argument("--stage", type=str, default="fusion", choices=["encoder", "fusion"])
    parser.add_argument("--class_id", type=int, default=-1, help="ID classifier target; -1 uses the predicted class")
    parser.add_argument("--target_layer", type=str, default="auto")
    return parser.parse_args()


def _load_model(cfg, device, weights_path: str):
    import torch

    from models.mv_srbf import MVSRBF

    model = MVSRBF(cfg).to(device)
    model.eval()
    if not weights_path:
        print("Grad-CAM is running without a checkpoint; overlays use the initialized model.")
        return model
    path = os.path.abspath(weights_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Grad-CAM checkpoint not found: {path}")
    print(f"Loading Grad-CAM checkpoint: {path}")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    print(f"Loaded Grad-CAM checkpoint: {path}")
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


def _iter_samples(loader: Iterable[Dict], max_samples: int):
    seen = 0
    for batch in loader:
        batch_size = int(batch["labels"].shape[0])
        for idx in range(batch_size):
            yield {
                "modalities": {k: v[idx : idx + 1] for k, v in batch["modalities"].items()},
                "aux_indices": {int(k): v[idx : idx + 1] for k, v in batch.get("aux_indices", {}).items()},
                "label": int(batch["labels"][idx]),
                "camid": int(batch["camids"][idx]) if "camids" in batch else 0,
            }
            seen += 1
            if seen >= max_samples:
                return


def _norm_params(norm_cfg: Dict, modal_key: str):
    item = norm_cfg.get(modal_key, norm_cfg.get("default", {}))
    return item.get("mean"), item.get("std")


def main() -> None:
    args = _parse_args()

    import numpy as np
    import torch
    from PIL import Image

    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as exc:
        raise ImportError(
            "tools/vis_grad_cam.py requires pytorch-grad-cam. "
            "Install it with `pip install grad-cam` in the active environment."
        ) from exc

    from data.dataloaders import build_dataloaders
    from utils.config import apply_parameter_overrides, get_config
    from utils.reproducibility import set_seed
    from visualization.grad_cam import (
        MVSRBFGradCAMWrapper,
        build_token_reshape_transform,
        infer_token_hw,
        resolve_grad_cam_target_layer,
        tensor_to_rgb_float,
    )

    cfg = get_config(os.path.abspath(args.config))
    apply_parameter_overrides(cfg, args.parameter)
    set_seed(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_dataloaders(cfg)
    loader_key = {"query": "query_loader", "gallery": "gallery_loader", "train_eval": "train_eval_loader"}[args.split]
    if loader_key not in loaders:
        raise ValueError(f"Split {args.split!r} is not available in the current dataloader config.")
    loader = loaders[loader_key]

    model = _load_model(cfg, device, args.weights)
    modal_keys = list(model.modal_keys)
    modal_names = [cfg["dataset"].get("modalities", {}).get(k, k) for k in modal_keys]
    selected_modalities = _resolve_modalities(args.modality, modal_keys, modal_names)
    norm_cfg = cfg.get("input", {}).get("normalize", {})

    os.makedirs(args.out_dir, exist_ok=True)
    for sample_idx, sample in enumerate(_iter_samples(loader, args.max_samples)):
        fixed_modalities = {k: v.to(device) for k, v in sample["modalities"].items()}
        aux_indices = {int(k): v.to(device) for k, v in sample["aux_indices"].items()}
        for modal_key in selected_modalities:
            input_tensor = fixed_modalities[modal_key]
            target_layer = resolve_grad_cam_target_layer(model, args.target_layer, modal_key=modal_key)
            wrapper = MVSRBFGradCAMWrapper(
                model,
                modal_key=modal_key,
                fixed_modalities=fixed_modalities,
                aux_indices=aux_indices,
                stage=args.stage,
            ).to(device)
            wrapper.eval()
            with torch.no_grad():
                logits = wrapper(input_tensor)
                target_class = int(logits.argmax(dim=1).item()) if args.class_id < 0 else int(args.class_id)

            spatial_hw = infer_token_hw(model, input_tensor, modal_key=modal_key)
            reshape_transform = build_token_reshape_transform(spatial_hw)
            targets = [ClassifierOutputTarget(target_class)]
            with GradCAM(model=wrapper, target_layers=[target_layer], reshape_transform=reshape_transform) as cam:
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

            mean, std = _norm_params(norm_cfg, modal_key)
            rgb = tensor_to_rgb_float(input_tensor[0], mean=mean, std=std).astype(np.float32)
            overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
            modal_name = cfg["dataset"].get("modalities", {}).get(modal_key, modal_key)
            filename = (
                f"{args.split}_sample{sample_idx:04d}_{modal_name}_{args.stage}"
                f"_pid{sample['label']}_target{target_class}.png"
            )
            path = os.path.join(args.out_dir, filename)
            Image.fromarray(overlay).save(path)
            print(f"Saved Grad-CAM: {path}")


if __name__ == "__main__":
    main()
