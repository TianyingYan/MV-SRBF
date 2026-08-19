#!/usr/bin/env python3
"""Run any-modal-to-any-modal MV-SRBF inference from a saved checkpoint."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML")
    parser.add_argument("--weights", type=str, default="", help="Checkpoint path; defaults to output_dir/checkpoints/best_train_loss.pth")
    parser.add_argument(
        "--parameter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override or add YAML fields. Example: --parameter inference.feature_stage=recover",
    )
    parser.add_argument(
        "--force-load",
        action="store_true",
        help="Load only checkpoint tensors that match the current model by name and shape.",
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Run inference without loading a checkpoint. This is only useful for pipeline smoke tests.",
    )
    args = parser.parse_args()

    import torch

    from data.dataloaders import build_dataloaders
    from engine.trainer import MVSRBFTrainer
    from utils.config import apply_parameter_overrides, archive_run_config, get_config
    from utils.reproducibility import set_seed

    cfg = get_config(os.path.abspath(args.config), include_metadata=True)
    apply_parameter_overrides(cfg, args.parameter)
    set_seed(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", True)))
    device_cfg = cfg.get("model", {}).get("device", {})
    if device_cfg.get("mode", "cuda_single") == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", int(device_cfg.get("ids", [0])[0]))

    # At test/inference time, pretrained and pretrained_weights_path are always
    # ignored because the project checkpoint overwrites all backbone weights.
    backbone_cfg = cfg.get("model", {}).get("encoder", {}).get("backbone", {})
    _pretrained_was = backbone_cfg.get("pretrained", False)
    _pwp_was = backbone_cfg.get("pretrained_weights_path", "")
    backbone_cfg["pretrained"] = False
    backbone_cfg["pretrained_weights_path"] = ""
    if _pretrained_was or _pwp_was:
        print(
            "Inference mode: backbone pretrained and pretrained_weights_path are ignored; "
            "all weights will be loaded from the project checkpoint."
        )

    loaders = build_dataloaders(cfg)
    archive_run_config(cfg, "test")
    trainer = MVSRBFTrainer(cfg, device)
    weights = os.path.abspath(args.weights) if args.weights else None
    try:
        trainer.test(
            loaders["query_loader"],
            loaders["gallery_loader"],
            weights_path=weights,
            load_checkpoint=not args.allow_random_init,
            force_load=args.force_load,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        message = str(exc)
        if message.startswith("Checkpoint is incompatible") or message.startswith("Inference checkpoint not found"):
            raise SystemExit(message) from None
        raise


if __name__ == "__main__":
    main()
