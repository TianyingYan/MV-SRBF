#!/usr/bin/env python3
"""Train MV-SRBF after activating the target environment.

Multi-GPU (DDP only, no DataParallel):
  torchrun --standalone --nproc_per_node=2 tools/train.py --config configs/experiments/rgbnt100.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

# repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _resolve_breakpoint_path(cfg, selection: str) -> str:
    """Map a --breakpoint selection to a checkpoint file under output_dir/checkpoints.

    Accepted selections:
      * ``latest`` / ``latest.pth`` -> ``output_dir/checkpoints/latest.pth`` (default).
      * ``best`` / ``best_train_loss`` / ``best_train_loss.pth`` ->
        ``output_dir/checkpoints/best_train_loss.pth``.
      * an explicit existing path to a ``.pth`` checkpoint.
    """
    raw = str(selection).strip()
    key = raw.lower()
    ckpt_dir = os.path.join(cfg.get("output_dir", "logs/run"), "checkpoints")
    aliases = {
        "latest": "latest.pth",
        "latest.pth": "latest.pth",
        "best": "best_train_loss.pth",
        "best_train_loss": "best_train_loss.pth",
        "best_train_loss.pth": "best_train_loss.pth",
    }
    if key in aliases:
        return os.path.abspath(os.path.join(ckpt_dir, aliases[key]))
    return os.path.abspath(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML")
    parser.add_argument(
        "--parameter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override or add YAML fields. Example: --parameter solver.max_epochs=10",
    )
    parser.add_argument(
        "-B",
        "--breakpoint",
        dest="breakpoint",
        nargs="?",
        const="latest",
        default=None,
        help=(
            "Load a project checkpoint under "
            "output_dir/checkpoints. Pass -B alone or '-B latest' for latest.pth "
            "(default), '-B best' for best_train_loss.pth, or '-B <path.pth>' for an "
            "explicit checkpoint. By default all saved training state is restored. "
            "Use --no-memory to load model weights only and restart training from zero."
        ),
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help=(
            "With --breakpoint, load only the project model weights. Ignore saved "
            "optimizer, scheduler, AMP, epoch/update, best-loss, early-stop, and "
            "other trainer state, then start from epoch 1 using the current YAML."
        ),
    )
    parser.add_argument(
        "--force-load",
        action="store_true",
        help="With --no-memory, load only checkpoint tensors that match the current model by name and shape.",
    )
    args = parser.parse_args()
    if args.no_memory and args.breakpoint is None:
        parser.error("--no-memory requires -B/--breakpoint.")
    if args.force_load and not args.no_memory:
        parser.error("--force-load is only supported together with -B/--breakpoint and --no-memory.")

    from utils.config import apply_parameter_overrides, archive_run_config, get_config

    cfg_path = os.path.abspath(args.config)
    cfg = get_config(cfg_path, include_metadata=True)
    apply_parameter_overrides(cfg, args.parameter)

    resume_from = None
    if args.breakpoint is not None:
        resume_from = _resolve_breakpoint_path(cfg, args.breakpoint)
        # On resume the project checkpoint provides all backbone weights, so backbone
        # pretrained / pretrained_weights_path are ignored (same policy as inference).
        backbone_cfg = cfg.get("model", {}).get("encoder", {}).get("backbone", {})
        _pretrained_was = backbone_cfg.get("pretrained", False)
        _pwp_was = backbone_cfg.get("pretrained_weights_path", "")
        backbone_cfg["pretrained"] = False
        backbone_cfg["pretrained_weights_path"] = ""
        mode_text = "loading model weights without training memory" if args.no_memory else "restoring full training state"
        print(f"Breakpoint training requested: {mode_text} from {resume_from}.")
        if _pretrained_was or _pwp_was:
            print(
                "Breakpoint training: backbone pretrained and pretrained_weights_path "
                "are ignored; weights are restored from the breakpoint checkpoint."
            )

    import torch
    import torch.distributed as dist

    mode = cfg.get("model", {}).get("device", {}).get("mode", "cuda_single")
    backend_cfg = str(cfg.get("model", {}).get("device", {}).get("dist_backend", "auto")).lower()
    if backend_cfg == "auto":
        dist_backend = "nccl" if torch.cuda.is_available() and dist.is_nccl_available() and os.name != "nt" else "gloo"
    else:
        dist_backend = backend_cfg

    if mode == "cpu":
        device = torch.device("cpu")
        local_rank = 0
    elif mode == "cuda_ddp":
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if world > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if not dist.is_initialized():
                dist.init_process_group(backend=dist_backend)
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            local_rank = 0
            device = torch.device("cuda", int(cfg.get("model", {}).get("device", {}).get("ids", [0])[0]))
    else:
        local_rank = 0
        device = torch.device("cuda", int(cfg.get("model", {}).get("device", {}).get("ids", [0])[0]))
    from utils.reproducibility import set_seed

    set_seed(
        int(cfg.get("seed", 42)),
        deterministic=bool(cfg.get("deterministic", True)),
    )

    from data.dataloaders import build_dataloaders
    from engine.trainer import MVSRBFTrainer

    loaders = build_dataloaders(cfg)
    global_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if global_rank == 0:
        archive_run_config(cfg, "train")
    trainer = MVSRBFTrainer(
        cfg,
        device,
        local_rank=local_rank,
        resume_from=resume_from,
        resume_memory=not args.no_memory,
        resume_force_load=args.force_load,
    )
    trainer.train(
        loaders["train_loader"],
        loaders.get("query_loader"),
        loaders.get("gallery_loader"),
        train_eval_loader=loaders.get("train_eval_loader"),
    )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
