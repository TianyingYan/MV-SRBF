"""Epoch-based MV-SRBF trainer for CPU, single GPU, and DDP."""
from __future__ import annotations

import os
import math
import time
import json
import hashlib
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from data.transforms.random_erasing import RandomErasingBatchAugment
from evaluation.any_modal_eval import evaluate_protocol
from evaluation.modality_utils import availability_to_cls_mask
from evaluation.postprocess import maybe_normalize_features
from models.mv_srbf import MVSRBF
from utils.checkpoint import save_checkpoint
from utils.lr_scheduler import build_lr_scheduler
from utils.model_complexity import count_parameters, format_large_number, profile_full_modal_flops
from utils.tracking import finish as tracker_finish
from utils.tracking import init_tracker, log_event, log_scalars


class MVSRBFTrainer:
    _LOSS_WEIGHT_KEYS = {
        "encoding_ce",
        "encoding_triplet",
        "encoding_center",
        "fusion_ce",
        "fusion_triplet",
        "fusion_center",
        "cross_triplet",
        "cross_center",
        "ortho_intra",
        "ortho_inter",
        "recovery",
    }
    _LR_MODULE_KEYS = {"backbone", "encoder", "modal_heads", "recovery", "fusion", "other", "default"}

    def __init__(
        self,
        cfg: Dict[str, Any],
        device: torch.device,
        local_rank: int = 0,
        resume_from: str | None = None,
        resume_memory: bool = True,
        resume_force_load: bool = False,
    ):
        self.cfg = cfg
        self.device = device
        self.local_rank = local_rank
        self.resume_path = resume_from
        self.resume_requested = bool(resume_from)
        self.resume_memory = bool(resume_memory)
        self.resume_force_load = bool(resume_force_load)
        self._resume_state: Dict[str, Any] | None = None
        self.is_ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        self.world_size = dist.get_world_size() if self.is_ddp else 1
        self.rank = dist.get_rank() if self.is_ddp else 0

        self.model = MVSRBF(cfg).to(device)
        if self.is_ddp:
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        from utils.optimizer_factory import build_optimizer

        self.optimizer = build_optimizer(cfg, self._unwrap_model())
        sol = cfg["solver"]
        if "gradient_accumulation_steps" in sol:
            raise ValueError("solver.gradient_accumulation_steps is not supported. Each mini-batch performs one optimizer update.")
        self.amp_enabled = bool(sol.get("amp_enabled", True)) and device.type == "cuda"
        amp_cfg = sol.get("amp", {})
        scaler_kwargs = {
            key: amp_cfg[key]
            for key in ("init_scale", "growth_factor", "backoff_factor", "growth_interval")
            if key in amp_cfg
        }
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled, **scaler_kwargs)
        lr_sched_cfg = sol.get("lr_scheduler", {})
        self.configured_max_epochs = int(sol.get("max_epochs", 120))
        self.max_epochs = self.configured_max_epochs
        self.warmup_updates = max(0, int(sol.get("warmup_updates", lr_sched_cfg.get("warmup_updates", 0))))
        self.scheduler = build_lr_scheduler(cfg, self.optimizer)

        self.log_period = int(sol.get("log_period", 50))
        # Per-component weighted loss contributions, surfaced by _compute_loss and
        # accumulated over each logging window so individual losses can be logged,
        # printed, and visualized (to observe cooperative vs conflicting effects).
        self._last_loss_components: Dict[str, torch.Tensor] = {}
        self._comp_log_sums: Dict[str, torch.Tensor] = {}
        self._comp_log_n: int = 0
        ckpt_cfg = sol.get("checkpoint", {})
        self.ckpt_period = int(ckpt_cfg.get("period_iters", 0))
        self.keep_last_snapshots = int(ckpt_cfg.get("keep_last", 10))
        self.save_snapshots = bool(ckpt_cfg.get("save_snapshots", False))
        early_cfg = sol.get("early_stop", {})
        self.early_stop_cfg = early_cfg
        self.early_stop_enabled = bool(early_cfg.get("enabled", True))
        self.early_stop_check_interval_iters = 0
        self.early_stop_patience_checks = int(early_cfg.get("patience_checks", 15))
        self.early_stop_min_delta = float(early_cfg.get("min_delta", 0.0))

        aug_cfg = cfg.get("input", {}).get("train_augmentation", {}).get("batch_augment", {})
        self.batch_aug = RandomErasingBatchAugment(aug_cfg) if aug_cfg.get("enable", False) else None
        train_eval_cfg = cfg.get("dataloader", {}).get("train_eval", {})
        self.train_eval_period_epochs = max(1, int(train_eval_cfg.get("period_epochs", 1)))

        self.output_dir = cfg.get("output_dir", "logs/run")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        self.best_train_loss = float("inf")
        self.best_iter = 0
        self.best_epoch = 0
        self.best_stop_loss = float("inf")
        self.best_stop_iter = 0
        self.bad_stop_checks = 0
        self.global_iter = 0
        self.global_update = 0
        self.epochs_completed = 0
        self.amp_skipped_updates = 0
        self.last_amp_scale_before = None
        self.last_amp_scale_after = None
        self.stop_window_loss = 0.0
        self.stop_window_n = 0
        self.snapshot_paths = deque()
        self.checkpoints_initialized = False
        if "soft" + "_stage" in sol:
            raise ValueError("Staged training is not supported by this training pipeline.")

    def _unwrap_model(self) -> nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _checkpoint_metadata(self) -> Dict[str, Any]:
        model_cfg = self.cfg.get("model", {})
        return {
            "model_name": model_cfg.get("name", "MVSRBF"),
            "backbone": dict(model_cfg.get("encoder", {}).get("backbone", {})),
            "feature_dim": int(getattr(self._unwrap_model(), "feature_dim", 0)),
            "modalities": list(self._unwrap_model().modal_keys),
        }

    @staticmethod
    def _validate_mapping_keys(values: Dict[str, Any], allowed: set[str], context: str) -> None:
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown {context} key(s): {unknown}. Allowed keys: {sorted(allowed)}.")

    def _loss_group_weights(self) -> Dict[str, float]:
        loss_cfg = self.cfg.get("model", {}).get("losses", {})
        weights = dict(loss_cfg.get("weights", {}))
        self._validate_mapping_keys(weights, self._LOSS_WEIGHT_KEYS, "model.losses.weights")
        return {str(k): float(v) for k, v in weights.items()}

    def _loss_group_weight(self, weights: Dict[str, float], key: str) -> float:
        return float(weights.get(key, 1.0))

    def _module_lr_summary(self) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        for group in self.optimizer.param_groups:
            if group.get("no_decay", False):
                continue
            module_name = str(group.get("module_name", "backbone" if group.get("is_backbone") else "other"))
            summary.setdefault(module_name, float(group["lr"]))
        return summary

    def _current_lr_metrics(self) -> Dict[str, float]:
        """Expose all active learning rates instead of only param group 0."""
        metrics: Dict[str, float] = {}
        module_lrs: Dict[str, float] = {}
        first_lr = None
        for idx, group in enumerate(self.optimizer.param_groups):
            lr = float(group["lr"])
            if first_lr is None:
                first_lr = lr
            module_name = str(group.get("module_name", "backbone" if group.get("is_backbone") else "other"))
            decay_tag = "no_decay" if bool(group.get("no_decay", False)) else "decay"
            metrics[f"train/lr_group/{idx}_{module_name}_{decay_tag}"] = lr
            if not bool(group.get("no_decay", False)):
                module_lrs.setdefault(module_name, lr)
        metrics["train/lr"] = float(first_lr or 0.0)
        for module_name, lr in module_lrs.items():
            metrics[f"train/lr_module/{module_name}"] = lr
        return metrics

    # Canonical display order for LR groups: pipeline order (module_lrs prefixes), then "other".
    _LR_GROUP_ORDER = ("backbone", "encoder", "modal_heads", "recovery", "fusion", "other")

    @classmethod
    def _ordered_lr_items(cls, summary: Dict[str, float]):
        order = cls._LR_GROUP_ORDER
        return sorted(
            summary.items(),
            key=lambda item: (order.index(item[0]), item[0]) if item[0] in order else (len(order), item[0]),
        )

    def _lr_text(self) -> str:
        summary = self._module_lr_summary()
        if not summary and self.optimizer.param_groups:
            summary = {"group0": float(self.optimizer.param_groups[0]["lr"])}
        return "[" + ", ".join(f"{k}={v:.6g}" for k, v in self._ordered_lr_items(summary)) + "]"

    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        modalities = {k: v.to(self.device, non_blocking=True) for k, v in batch["modalities"].items()}
        labels = batch["labels"].to(self.device, non_blocking=True)
        aux_indices = {
            int(k): v.to(self.device, non_blocking=True)
            for k, v in batch.get("aux_indices", {}).items()
        }

        if self.batch_aug is not None:
            modalities = self.batch_aug(modalities)

        mcfg = self.cfg["model"]["mv_srbf"]
        loss_cfg = self.cfg["model"].get("losses", {})
        cross_cfg = loss_cfg.get("cross_modal", {})
        # Per-term weight is the single magnitude control (no per-loss-type scales).
        group_weights = self._loss_group_weights()
        recovery_weight = self._loss_group_weight(group_weights, "recovery")
        ortho_intra_weight = self._loss_group_weight(group_weights, "ortho_intra")
        ortho_inter_weight = self._loss_group_weight(group_weights, "ortho_inter")
        enc_ce_weight = self._loss_group_weight(group_weights, "encoding_ce")
        enc_tri_weight = self._loss_group_weight(group_weights, "encoding_triplet")
        enc_center_weight = self._loss_group_weight(group_weights, "encoding_center")
        fuse_ce_weight = self._loss_group_weight(group_weights, "fusion_ce")
        fuse_tri_weight = self._loss_group_weight(group_weights, "fusion_triplet")
        fuse_center_weight = self._loss_group_weight(group_weights, "fusion_center")
        cross_tri_weight = self._loss_group_weight(group_weights, "cross_triplet")
        cross_center_weight = self._loss_group_weight(group_weights, "cross_center",)
        cross_triplet_cfg = cross_cfg.get("triplet", {})
        cross_center_cfg = cross_cfg.get("center", {})
        enc_ce_active = enc_ce_weight != 0.0
        enc_tri_active = enc_tri_weight != 0.0
        enc_center_active = enc_center_weight != 0.0
        fuse_ce_active = fuse_ce_weight != 0.0
        fuse_tri_active = fuse_tri_weight != 0.0
        fuse_center_active = fuse_center_weight != 0.0
        recovery_arch_enabled = bool(mcfg.get("recovery", {}).get("enabled", True))
        recovery_loss_active = recovery_arch_enabled and recovery_weight != 0.0
        cross_tri_active = cross_tri_weight != 0.0 and bool(cross_triplet_cfg.get("enabled", False))
        cross_center_active = cross_center_weight != 0.0 and bool(cross_center_cfg.get("enabled", False))
        ortho_intra_active = ortho_intra_weight != 0.0
        ortho_inter_active = ortho_inter_weight != 0.0
        use_fusion = fuse_ce_active or fuse_tri_active or fuse_center_active
        use_recovery = recovery_loss_active or use_fusion or ortho_inter_active
        any_loss_active = (
            enc_ce_active
            or enc_tri_active
            or enc_center_active
            or fuse_ce_active
            or fuse_tri_active
            or fuse_center_active
            or recovery_loss_active
            or cross_tri_active
            or cross_center_active
            or ortho_intra_active
            or ortho_inter_active
        )
        if not any_loss_active:
            raise ValueError("No active training loss. Check model.losses.weights in the YAML config.")

        with torch.amp.autocast("cuda", enabled=self.amp_enabled):
            out = self.model(
                modalities,
                labels,
                aux_indices=aux_indices,
                use_recovery=use_recovery,
                use_fusion=use_fusion,
                loss_options={
                    "encoding": enc_ce_active or enc_tri_active or enc_center_active,
                    "enc_ce": enc_ce_active,
                    "enc_tri": enc_tri_active,
                    "enc_center": enc_center_active,
                    "fusion": use_fusion,
                    "fuse_ce": fuse_ce_active,
                    "fuse_tri": fuse_tri_active,
                    "fuse_center": fuse_center_active,
                    "cross_modal": cross_tri_active or cross_center_active,
                    "cross_tri": cross_tri_active,
                    "cross_center": cross_center_active,
                    "ortho_intra": ortho_intra_active,
                    "ortho_inter": ortho_inter_active,
                },
            )
            ld = out["loss_dict"]
            # Weighted contribution of each term (weight is the single magnitude knob;
            # these sum to the total objective).
            contrib = {
                "enc_ce": enc_ce_weight * ld["enc_ce"],
                "enc_tri": enc_tri_weight * ld["enc_tri"],
                "enc_center": enc_center_weight * ld["enc_center"],
                "fuse_ce": fuse_ce_weight * ld["fuse_ce"],
                "fuse_tri": fuse_tri_weight * ld["fuse_tri"],
                "fuse_center": fuse_center_weight * ld["fuse_center"],
                "cross_tri": cross_tri_weight * ld["cross_tri"],
                "cross_center": cross_center_weight * ld["cross_center"],
                "rec": recovery_weight * ld["rec"],
                "ortho_intra": ortho_intra_weight * ld["ortho_intra"],
                "ortho_inter": ortho_inter_weight * ld["ortho_inter"],
            }
            loss = sum(contrib.values())

        # Surface only the active terms (nonzero weight) plus raw recovery sub-losses,
        # detached, so the training loop can log/print each loss without holding the graph.
        # `rec` is placed after the ortho terms so it groups with its raw recovery
        # diagnostics (rec_cls_mse) appended below.
        active_flags = {
            "enc_ce": enc_ce_active, "enc_tri": enc_tri_active, "enc_center": enc_center_active,
            "fuse_ce": fuse_ce_active, "fuse_tri": fuse_tri_active, "fuse_center": fuse_center_active,
            "cross_tri": cross_tri_active, "cross_center": cross_center_active,
            "ortho_intra": ortho_intra_active, "ortho_inter": ortho_inter_active,
            "rec": recovery_loss_active,
        }
        components = {name: contrib[name].detach() for name, flag in active_flags.items() if flag}
        if recovery_loss_active:
            # Raw recovery diagnostics (unweighted) to observe the recovery branch itself.
            for raw_key in ("rec_cls_mse",):
                if raw_key in ld:
                    components[raw_key] = ld[raw_key].detach()
        self._last_loss_components = components
        return loss

    def _accumulate_loss_components(self) -> None:
        """Sum the latest per-component losses into the current logging window (on-device).

        Only rank 0 accumulates, matching where the window is drained and logged.
        """
        if self.rank != 0:
            return
        for name, value in self._last_loss_components.items():
            if name in self._comp_log_sums:
                self._comp_log_sums[name] = self._comp_log_sums[name] + value
            else:
                self._comp_log_sums[name] = value
        if self._last_loss_components:
            self._comp_log_n += 1

    def _drain_loss_components(self) -> Dict[str, float]:
        """Return mean per-component losses over the window and reset the accumulator."""
        if self._comp_log_n <= 0:
            self._comp_log_sums = {}
            return {}
        out = {name: float(total) / self._comp_log_n for name, total in self._comp_log_sums.items()}
        self._comp_log_sums = {}
        self._comp_log_n = 0
        return out

    def _step_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Run one immediate optimizer update for tests and debugging utilities."""
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._compute_loss(batch)
        self.grad_scaler.scale(loss).backward()
        stepped = self._optimizer_step()
        if stepped:
            self.global_update += 1
        self.global_iter += 1
        return loss.detach()

    def _optimizer_step(self) -> bool:
        """Step the optimizer and report whether AMP skipped the update."""
        self.last_amp_scale_before = None
        self.last_amp_scale_after = None
        if not self.grad_scaler.is_enabled():
            self.optimizer.step()
            return True
        scale_before = self.grad_scaler.get_scale()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        scale_after = self.grad_scaler.get_scale()
        self.last_amp_scale_before = scale_before
        self.last_amp_scale_after = scale_after
        return scale_after >= scale_before

    def train_epoch(self, loader, epoch: int) -> Dict[str, float]:
        self.model.train()
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        start_time = time.time()
        totals = {"loss": 0.0, "n": 0, "samples": 0}
        stop_now = False
        epoch_iters = len(loader)
        epoch_updates = 0
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            loss_t = self._compute_loss(batch)
            loss_value = float(loss_t)
            self._accumulate_loss_components()
            self.grad_scaler.scale(loss_t).backward()
            self.global_iter += 1
            totals["loss"] += loss_value
            totals["n"] += 1
            if "labels" in batch:
                totals["samples"] += int(batch["labels"].shape[0])

            stepped = self._optimizer_step()
            if not stepped:
                self.amp_skipped_updates += 1
                self.optimizer.zero_grad(set_to_none=True)
                if self.rank == 0:
                    scale_text = (
                        f" amp_scale={self.last_amp_scale_before:.6g}->{self.last_amp_scale_after:.6g};"
                        if self.last_amp_scale_before is not None and self.last_amp_scale_after is not None
                        else ""
                    )
                    print(
                        f"Skipped optimizer update at Epoch {epoch + 1} "
                        f"Iter {step + 1}/{epoch_iters} due to non-finite gradients; "
                        f"scheduler was not stepped;{scale_text} "
                        f"skipped_updates={self.amp_skipped_updates}."
                    )
                continue

            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_update += 1
            epoch_updates += 1
            update_loss = loss_value
            self.stop_window_loss += update_loss
            self.stop_window_n += 1

            if self.rank == 0 and self.ckpt_period > 0 and self.global_update % self.ckpt_period == 0:
                self._save_latest(self.global_update)
                if self.save_snapshots:
                    self._save_snapshot(self.global_update)

            if self.rank == 0 and (self.global_update == 1 or self.global_update % self.log_period == 0):
                lr_metrics = self._current_lr_metrics()
                avg_loss = totals["loss"] / max(1, totals["n"])
                best_text = (
                    f"{self.best_train_loss:.4f}@epoch{self.best_epoch}"
                    if self.best_epoch > 0
                    else "pending"
                )
                loss_components = self._drain_loss_components()
                log_scalars(
                    {
                        "train/loss": update_loss,
                        "train/epoch_avg_loss": avg_loss,
                        "train/global_iter": self.global_iter,
                        **{f"train/loss_components/{name}": value for name, value in loss_components.items()},
                        **lr_metrics,
                    },
                    self.global_update,
                )
                comp_text = (
                    " | losses=[" + ", ".join(f"{name}={value:.4f}" for name, value in loss_components.items()) + "]"
                    if loss_components
                    else ""
                )
                print(
                    f"Epoch {epoch + 1}/{self.max_epochs}, "
                    f"Iter {step + 1}/{epoch_iters}, "
                    f"GlobalUpdate {self.global_update}, "
                    f"loss={update_loss:.4f}, "
                    f"best_epoch_avg={best_text}"
                    f"{comp_text}"
                    f" | lr={self._lr_text()}"
                )

            if self.rank == 0 and self._should_check_early_stop_window(step + 1, epoch_iters):
                window_loss = self.stop_window_loss / max(1, self.stop_window_n)
                stop_now = self._update_early_stop_by_window(window_loss)
                log_scalars(
                    {
                        "train/early_stop_window_loss": window_loss,
                        "train/early_stop_bad_checks": self.bad_stop_checks,
                    },
                    self.global_update,
                )
                print(
                    f"EarlyStopCheck update={self.global_update} "
                    f"window_avg_loss={window_loss:.4f} "
                    f"best_window_avg={self.best_stop_loss:.4f}@update{self.best_stop_iter} "
                    f"bad_checks={self.bad_stop_checks}/{self.early_stop_patience_checks}"
                )
                self.stop_window_loss = 0.0
                self.stop_window_n = 0
            elif self.rank != 0:
                stop_now = False
            if self.is_ddp:
                stop_flag = torch.tensor([int(stop_now)], device=self.device)
                dist.broadcast(stop_flag, src=0)
                stop_now = bool(stop_flag.item())
            if stop_now:
                break

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = max(1e-9, time.time() - start_time)
        global_samples = totals["samples"] * self.world_size
        return {
            "loss": totals["loss"] / max(1, totals["n"]),
            "iters": float(totals["n"]),
            "updates": float(epoch_updates),
            "samples": float(global_samples),
            "time_sec": float(elapsed),
            "time_per_iter_sec": float(elapsed / max(1, totals["n"])),
            "time_per_update_sec": float(elapsed / max(1, epoch_updates)),
            "samples_per_sec": float(global_samples / elapsed),
            "stop": float(stop_now),
        }

    def _should_check_early_stop_window(self, step: int, epoch_iters: int) -> bool:
        if not self.early_stop_enabled or self.global_update <= self.warmup_updates:
            return False
        interval = max(1, int(self.early_stop_check_interval_iters))
        return self.stop_window_n > 0 and (self.global_update % interval == 0 or step >= epoch_iters)

    def _update_early_stop_by_window(self, loss: float) -> bool:
        if loss < self.best_stop_loss - self.early_stop_min_delta:
            self.best_stop_loss = loss
            self.best_stop_iter = self.global_update
            self.bad_stop_checks = 0
            return False
        self.bad_stop_checks += 1
        return self.bad_stop_checks >= self.early_stop_patience_checks

    def _checkpoint_path(self, filename: str) -> str:
        return os.path.join(self.output_dir, "checkpoints", filename)

    def _save_checkpoint_file(self, filename: str, step: int) -> str:
        if self.rank != 0:
            return ""
        path = self._checkpoint_path(filename)
        save_checkpoint(
            path,
            self._unwrap_model(),
            self.optimizer,
            step,
            metadata=self._checkpoint_metadata(),
            extra=self._trainer_state_dict(),
        )
        return path

    def _trainer_state_dict(self) -> Dict[str, Any]:
        """Collect resumable trainer state for breakpoint training.

        The payload restores the optimizer-update bookkeeping, best-checkpoint and
            early-stop tracking, AMP scaler and LR scheduler state so that
            ``--breakpoint`` continues a run exactly where it stopped.
        Per-rank RNG state is deliberately not stored: samplers and DataLoader
        workers are reseeded per epoch from the restored epoch index, which keeps
        DDP resume deterministic without replaying a single rank's RNG everywhere.
        """
        state: Dict[str, Any] = {
            "version": 1,
            "global_iter": int(self.global_iter),
            "global_update": int(self.global_update),
            "epochs_completed": int(self.epochs_completed),
            "max_epochs": int(self.max_epochs),
            "best_train_loss": float(self.best_train_loss),
            "best_iter": int(self.best_iter),
            "best_epoch": int(self.best_epoch),
            "best_stop_loss": float(self.best_stop_loss),
            "best_stop_iter": int(self.best_stop_iter),
            "bad_stop_checks": int(self.bad_stop_checks),
            "amp_skipped_updates": int(self.amp_skipped_updates),
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        if self.amp_enabled:
            state["grad_scaler"] = self.grad_scaler.state_dict()
        return {"trainer_state": state}

    def _resume_load_payload(self) -> None:
        """Load a project checkpoint using full-resume or weights-only semantics.

        Runs on every rank so all DDP replicas start from identical model weights.
        Full resume also restores optimizer/trainer state. No-memory resume ignores
        every saved training state and starts from epoch/update zero under the
        current YAML configuration.
        """
        if not self.resume_requested or not self.resume_path:
            return
        if not os.path.isfile(self.resume_path):
            raise FileNotFoundError(
                f"Breakpoint checkpoint not found: {self.resume_path}. "
                "Train at least one epoch first, or point --breakpoint at an existing "
                "checkpoint such as latest.pth or best_train_loss.pth."
            )
        checkpoint = self._torch_load_checkpoint(self.resume_path)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise RuntimeError(
                f"Breakpoint checkpoint {self.resume_path} is not a trainer checkpoint "
                "(missing 'model' state). Use a checkpoint produced by tools/train.py."
            )
        self._load_model_state(
            checkpoint["model"],
            context="breakpoint checkpoint",
            force=self.resume_force_load,
        )
        if not self.resume_memory:
            self._resume_state = None
            self.checkpoints_initialized = False
            if self.rank == 0:
                force_text = " Force-load was used; incompatible tensors were skipped." if self.resume_force_load else ""
                print(
                    "Breakpoint no-memory mode: loaded model weights only; optimizer, "
                    "scheduler, AMP scaler, epoch/update counters, best metrics, "
                    "and early-stop state start from zero."
                    f"{force_text}"
                )
            return

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        state = checkpoint.get("trainer_state")
        if not isinstance(state, dict):
            if self.rank == 0:
                print(
                    "Breakpoint warning: checkpoint has no trainer_state block "
                    "(it predates breakpoint support). Resuming with model and "
                    "optimizer weights only; epoch/scheduler counters restart at zero."
                )
            self._resume_state = None
        else:
            self.global_iter = int(state.get("global_iter", 0))
            self.global_update = int(state.get("global_update", 0))
            self.epochs_completed = int(state.get("epochs_completed", 0))
            self.best_train_loss = float(state.get("best_train_loss", self.best_train_loss))
            self.best_iter = int(state.get("best_iter", 0))
            self.best_epoch = int(state.get("best_epoch", 0))
            self.best_stop_loss = float(state.get("best_stop_loss", self.best_stop_loss))
            self.best_stop_iter = int(state.get("best_stop_iter", 0))
            self.bad_stop_checks = int(state.get("bad_stop_checks", 0))
            self.amp_skipped_updates = int(state.get("amp_skipped_updates", 0))
            self._resume_state = state
        # The breakpoint already holds trained weights, so the standard zero-step
        # init must not overwrite latest.pth / best_train_loss.pth.
        self.checkpoints_initialized = True

    def _resume_restore_schedule(self) -> None:
        """Restore LR scheduler and AMP scaler after the scheduler is rebuilt."""
        if not self.resume_requested or not self.resume_memory or not isinstance(self._resume_state, dict):
            return
        sched_state = self._resume_state.get("scheduler")
        if sched_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(sched_state)
        scaler_state = self._resume_state.get("grad_scaler")
        if scaler_state is not None and self.amp_enabled:
            self.grad_scaler.load_state_dict(scaler_state)

    def _save_latest(self, step: int) -> None:
        self._save_checkpoint_file("latest.pth", step)

    def _save_best(self, step: int) -> None:
        self._save_checkpoint_file("best_train_loss.pth", step)

    def _save_snapshot(self, step: int) -> None:
        if self.keep_last_snapshots <= 0:
            return
        path = self._save_checkpoint_file(f"snapshot_iter_{step}.pth", step)
        if not path:
            return
        self.snapshot_paths.append(path)
        while len(self.snapshot_paths) > self.keep_last_snapshots:
            old = self.snapshot_paths.popleft()
            if os.path.isfile(old):
                os.remove(old)

    def _initialize_checkpoint_files(self) -> None:
        if self.rank != 0 or self.checkpoints_initialized:
            return
        self._save_latest(0)
        self._save_best(0)
        self.checkpoints_initialized = True

    def _save_best_by_train_loss(self, loss: float, iteration: int, epoch: int) -> bool:
        if self.rank != 0:
            return False
        if loss < self.best_train_loss - self.early_stop_min_delta:
            self.best_train_loss = loss
            self.best_iter = iteration
            self.best_epoch = epoch
            self._save_best(iteration)
            return True
        return False

    def _load_best_train_loss_checkpoint(self) -> None:
        path = self._checkpoint_path("best_train_loss.pth")
        if self.rank == 0 and os.path.isfile(path):
            checkpoint = self._torch_load_checkpoint(path)
            self._unwrap_model().load_state_dict(checkpoint["model"], strict=True)

    def _log_model_summary(self) -> None:
        """Print and log parameter count plus optional full-modal FLOPs."""
        if self.rank != 0:
            return
        summary_cfg = self.cfg.get("logging", {}).get("model_summary", {})
        if not bool(summary_cfg.get("enabled", True)):
            return

        model = self._unwrap_model()
        params = count_parameters(model)
        metrics = {
            "model/params_total": float(params["total"]),
            "model/params_trainable": float(params["trainable"]),
        }
        message = (
            f"Model summary: params total={format_large_number(params['total'])}, "
            f"trainable={format_large_number(params['trainable'])}"
        )
        if bool(summary_cfg.get("flops", True)):
            try:
                profile = profile_full_modal_flops(
                    model,
                    self.cfg,
                    self.device,
                    backend=str(summary_cfg.get("flops_backend", "fvcore")),
                    allow_fallback=bool(summary_cfg.get("allow_fallback", True)),
                )
                flops = int(profile["value"])
                backend = str(profile.get("backend", "unknown"))
                metrics["model/flops_full_modal"] = float(flops)
                metrics[f"model/flops_backend/{backend}"] = 1.0
                message += f", full-modal FLOPs={format_large_number(flops)} ({backend})"
                unsupported = profile.get("unsupported_ops", {})
                if unsupported:
                    message += f", unsupported_ops={len(unsupported)}"
                if profile.get("fallback_reason"):
                    message += f", fallback_reason={profile['fallback_reason']}"
            except Exception as exc:
                message += f", full-modal FLOPs=unavailable ({exc})"
        log_scalars(metrics, 0)
        print(message)

    def _archive_run_state(self) -> None:
        """Save exact run settings and model state metadata for later audits."""
        if self.rank != 0:
            return
        log_dir = Path(self.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        clean_cfg = {k: v for k, v in self.cfg.items() if not str(k).startswith("_")}
        config_text = yaml.safe_dump(clean_cfg, sort_keys=False, allow_unicode=True)
        model_text = str(self._unwrap_model())
        with open(log_dir / "effective_config.yaml", "w", encoding="utf-8") as handle:
            handle.write(config_text)
        with open(log_dir / "model_architecture.txt", "w", encoding="utf-8") as handle:
            handle.write(model_text)
            handle.write("\n")
        with open(log_dir / "optimizer_param_groups.json", "w", encoding="utf-8") as handle:
            json.dump(self._optimizer_group_summary(), handle, indent=2)
        with open(log_dir / "source_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(self._source_manifest(), handle, indent=2)
        print(
            "Run archive saved: "
            f"{log_dir / 'effective_config.yaml'}, "
            f"{log_dir / 'model_architecture.txt'}, "
            f"{log_dir / 'optimizer_param_groups.json'}, "
            f"{log_dir / 'source_manifest.json'}"
        )
        archive_cfg = self.cfg.get("logging", {}).get("archive", {})
        if bool(archive_cfg.get("print_config", True)):
            print("Effective configuration:\n" + config_text.rstrip())
        if bool(archive_cfg.get("print_model", True)):
            print("Model architecture:\n" + model_text)

    def _optimizer_group_summary(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for idx, group in enumerate(self.optimizer.param_groups):
            param_count = sum(int(p.numel()) for p in group.get("params", []))
            trainable_count = sum(int(p.numel()) for p in group.get("params", []) if p.requires_grad)
            rows.append(
                {
                    "index": idx,
                    "module_name": group.get("module_name"),
                    "is_backbone": bool(group.get("is_backbone", False)),
                    "no_decay": bool(group.get("no_decay", False)),
                    "lr": float(group.get("lr", 0.0)),
                    "base_lr": float(group.get("base_lr", group.get("lr", 0.0))),
                    "weight_decay": float(group.get("weight_decay", 0.0)),
                    "params": param_count,
                    "trainable_params": trainable_count,
                }
            )
        return rows

    def _source_manifest(self) -> Dict[str, Any]:
        root = Path(__file__).resolve().parents[1]
        suffixes = {".py", ".yaml", ".yml", ".md", ".toml", ".txt"}
        scan_entries = [
            "configs",
            "data",
            "docs",
            "engine",
            "evaluation",
            "losses",
            "models",
            "tools",
            "utils",
            "visualization",
            "tests",
            ".gitignore",
            "MANIFEST.in",
            "pyproject.toml",
            "README.md",
            "requirements.txt",
        ]
        files = []
        candidates: List[Path] = []
        for entry in scan_entries:
            path = root / entry
            if path.is_dir():
                candidates.extend(path.rglob("*"))
            elif path.exists():
                candidates.append(path)
        for path in sorted(candidates):
            if not path.is_file() or path.suffix.lower() not in suffixes or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(root).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            stat = path.stat()
            files.append(
                {
                    "path": rel,
                    "sha256": digest,
                    "size": int(stat.st_size),
                    "mtime": float(stat.st_mtime),
                }
            )
        return {
            "root": str(root),
            "created_time": time.time(),
            "files": files,
        }

    @torch.no_grad()
    def evaluate_train_accuracy(self, train_eval_loader, epoch: int) -> Dict[str, float]:
        """Evaluate classification accuracy on all training images without train-time augmentation."""
        if train_eval_loader is None:
            return {}
        if self.rank != 0:
            if self.is_ddp:
                dist.barrier()
            return {}

        model = self._unwrap_model()
        was_training = model.training
        model.eval()
        correct = {"fuse": 0}
        total = 0
        for key in model.modal_keys:
            correct[key] = 0

        for batch in train_eval_loader:
            modalities = {k: v.to(self.device, non_blocking=True) for k, v in batch["modalities"].items()}
            labels = batch["labels"].to(self.device, non_blocking=True)
            aux_indices = {
                int(k): v.to(self.device, non_blocking=True)
                for k, v in batch.get("aux_indices", {}).items()
            }
            out = model.classify_full_modalities(modalities, aux_indices=aux_indices)
            fuse_pred = out["fuse_logits"].argmax(dim=1)
            correct["fuse"] += int((fuse_pred == labels).sum().item())
            modal_logits = out["modal_logits"]
            for key in model.modal_keys:
                pred = modal_logits[key].argmax(dim=1)
                correct[key] += int((pred == labels).sum().item())
            total += int(labels.numel())

        if was_training:
            model.train()
        denom = max(1, total)
        metrics = {"train_acc/fuse": correct["fuse"] / denom}
        modality_names = self.cfg.get("dataset", {}).get("modalities", {})
        parts = [f"fuse={metrics['train_acc/fuse']:.4f}"]
        for key in model.modal_keys:
            acc = correct[key] / denom
            metrics[f"train_acc/{key}"] = acc
            name = modality_names.get(key, key)
            parts.append(f"{key}({name})={acc:.4f}")
        log_scalars(metrics, self.global_update)
        print(f"TrainAcc epoch {epoch}: " + ", ".join(parts))

        if self.is_ddp:
            dist.barrier()
        return metrics

    def _should_evaluate_train_accuracy(self, train_eval_loader, epoch: int) -> bool:
        if train_eval_loader is None:
            return False
        return epoch % self.train_eval_period_epochs == 0

    def _load_checkpoint_for_inference(
        self,
        weights_path: str | None = None,
        force: bool = False,
    ) -> bool:
        path = weights_path or self._checkpoint_path("best_train_loss.pth")
        if self.rank != 0:
            return False
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Inference checkpoint not found: {path}")
        checkpoint = self._torch_load_checkpoint(path)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        try:
            self._load_model_state(state, context="inference checkpoint", force=force)
        except RuntimeError:
            if force:
                raise
            ckpt_meta = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
            ckpt_backbone = self._infer_checkpoint_backbone_name(state, ckpt_meta)
            cfg_backbone = self.cfg.get("model", {}).get("encoder", {}).get("backbone", {}).get("name", "unknown")
            raise RuntimeError(
                "Checkpoint is incompatible with the current model config. "
                f"Config backbone={cfg_backbone!r}, checkpoint backbone={ckpt_backbone!r}. "
                "Train a new checkpoint after changing backbone, pass a matching --weights file, "
                "or use --allow-random-init only for pipeline smoke tests."
            ) from None
        return True

    def _load_model_state(self, state: Dict[str, Any], *, context: str, force: bool = False) -> None:
        """Load model tensors strictly, or force-load only compatible tensors."""
        model = self._unwrap_model()
        if not force:
            model.load_state_dict(state, strict=True)
            return
        if not isinstance(state, dict):
            raise RuntimeError(f"Cannot force-load {context}: checkpoint state is not a mapping.")
        current = model.state_dict()
        compatible = {}
        skipped = []
        for key, value in state.items():
            if key not in current:
                skipped.append((key, "missing_in_model"))
                continue
            if not hasattr(value, "shape"):
                skipped.append((key, f"non_tensor {type(value).__name__}"))
                continue
            if tuple(current[key].shape) != tuple(value.shape):
                skipped.append((key, f"shape {tuple(value.shape)} != {tuple(current[key].shape)}"))
                continue
            compatible[key] = value
        missing = [key for key in current.keys() if key not in compatible]
        model.load_state_dict(compatible, strict=False)
        print(
            f"Force-loaded {context}: matched={len(compatible)}, "
            f"skipped={len(skipped)}, missing_in_checkpoint={len(missing)}. "
            "Only same-name same-shape tensors were loaded."
        )
        if skipped:
            preview = ", ".join(f"{key} ({reason})" for key, reason in skipped[:8])
            suffix = " ..." if len(skipped) > 8 else ""
            print(f"Force-load skipped tensors: {preview}{suffix}")
        if not compatible:
            print(f"Warning: force-load found no compatible tensors in {context}; model remains initialized.")

    def _torch_load_checkpoint(self, path: str):
        try:
            return torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=self.device)

    def _infer_checkpoint_backbone_name(self, state: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        backbone = metadata.get("backbone", {}) if isinstance(metadata, dict) else {}
        if isinstance(backbone, dict) and backbone.get("name"):
            return str(backbone["name"])
        keys = set(state.keys()) if isinstance(state, dict) else set()
        if any(key.startswith("encoder.trunk.encoder.layers.") for key in keys):
            return "vision_transformer"
        if any(".trunk.encoder.layers." in key for key in keys if key.startswith("encoder.encoders.")):
            return "vision_transformer"
        if "encoder.trunk.class_token" in keys or "encoder.trunk.conv_proj.weight" in keys:
            return "vision_transformer"
        if any(
            key.startswith("encoder.encoders.") and (".trunk.class_token" in key or ".trunk.conv_proj.weight" in key)
            for key in keys
        ):
            return "vision_transformer"
        if any(key.startswith("encoder.trunk.4.") for key in keys):
            return "resnet_like"
        if any(key.startswith("encoder.encoders.") and ".trunk.4." in key for key in keys):
            return "resnet_like"
        return "unknown"

    @staticmethod
    def _batch_aux_matrix(batch: Dict[str, Any]) -> torch.Tensor:
        """Build a [B, A] aux matrix whose column index matches the dataset aux dim."""
        aux_indices = batch.get("aux_indices", {})
        if aux_indices:
            max_dim = max(int(dim) for dim in aux_indices.keys())
            first = next(iter(aux_indices.values()))
            cols = []
            for dim in range(max_dim + 1):
                value = aux_indices.get(dim)
                if value is None:
                    value = torch.full_like(first, -1)
                cols.append(value.detach().cpu())
            return torch.stack(cols, dim=1)
        return torch.empty((int(batch["labels"].shape[0]), 0), dtype=torch.long)

    @torch.no_grad()
    def _extract_any_modal_batch(
        self,
        batch: Dict[str, Any],
        role: str,
        kp=None,
        q_mod_mask: torch.Tensor | None = None,
        g_mod_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        model = self._unwrap_model()
        model.eval()
        modalities = {k: v.to(self.device, non_blocking=True) for k, v in batch["modalities"].items()}
        labels = batch["labels"].to(self.device, non_blocking=True)
        aux_indices = {
            int(k): v.to(self.device, non_blocking=True)
            for k, v in batch.get("aux_indices", {}).items()
        }
        availability = q_mod_mask if role == "query" else g_mod_mask
        if availability is None:
            availability = torch.ones(len(model.modal_keys), dtype=torch.bool, device=self.device)
        available_indices = torch.nonzero(availability, as_tuple=False).flatten().tolist()
        batch_size = labels.shape[0]
        mask = availability_to_cls_mask(batch_size, len(model.modal_keys), available_indices, self.device)
        infer = self.cfg.get("inference", {})
        from evaluation.feature_extraction import extract_fused_feature

        feat = extract_fused_feature(
            model,
            modalities,
            labels,
            mask,
            aux_indices=aux_indices,
            neck_feat=infer.get("neck_feat", "before"),
            feature_stage=infer.get("feature_stage", "fusion"),
            feature_aggregation=infer.get("feature_aggregation", "mean"),
        )
        feat = maybe_normalize_features(feat, bool(infer.get("feat_norm", True)))
        camids = batch.get("camids", torch.zeros_like(batch["labels"]))
        return {
            "feat": feat.detach().cpu(),
            "pids": batch["labels"].detach().cpu(),
            "camids": camids.detach().cpu(),
            "aux": self._batch_aux_matrix(batch),
        }

    def test(
        self,
        query_loader,
        gallery_loader,
        weights_path: str | None = None,
        load_checkpoint: bool = True,
        force_load: bool = False,
    ) -> Dict[str, float]:
        if self.rank != 0:
            return {}
        if load_checkpoint:
            self._load_checkpoint_for_inference(weights_path, force=force_load)
        else:
            print("Inference is running without loading a checkpoint; metrics are for a randomly initialized model.")
        infer = self.cfg.get("inference", {})
        max_rank = int(infer.get("default_num", 50))
        metric = infer.get("metric", "cosine")
        ranks = [int(r) for r in infer.get("ranks", [1, 5, 10])]
        num_modalities = len(self._unwrap_model().modal_keys)
        model = self._unwrap_model()
        metrics = evaluate_protocol(
            self._extract_any_modal_batch,
            query_loader,
            gallery_loader,
            num_modalities=num_modalities,
            device=self.device,
            metric=metric,
            max_rank=max_rank,
            infer_cfg=infer,
            modal_names=list(model.modal_keys),
        )
        macro_groups = metrics.get("macro", {})
        macro = macro_groups.get("symmetric_macro", {})
        cmc = macro.get("cmc", [])
        result = {
            "test/mAP_macro": float(macro.get("mAP", 0.0)),
        }
        for rank in ranks:
            result[f"test/Rank{rank}_macro"] = float(cmc[rank - 1]) if len(cmc) >= rank else 0.0
        log_scalars(result, self.best_iter)
        rank_text = ", ".join(f"Rank{r}={result[f'test/Rank{r}_macro']:.4f}" for r in ranks)
        print(f"Any-modal test: mAP@{max_rank}={result['test/mAP_macro']:.4f}, {rank_text}")
        self._write_inference_report(metrics, result, ranks, max_rank)
        return result

    def _write_inference_report(self, metrics: Dict[str, Any], result: Dict[str, float], ranks, default_num: int) -> None:
        if self.rank != 0:
            return
        import json

        infer_dir = os.path.join(self.output_dir, "inference")
        os.makedirs(infer_dir, exist_ok=True)

        def enrich_rows(rows):
            enriched = []
            for row in rows:
                item = dict(row)
                cmc_arr = item.get("cmc", [])
                for rank in ranks:
                    item[f"Rank{rank}"] = float(cmc_arr[rank - 1]) if len(cmc_arr) >= rank else 0.0
                enriched.append(item)
            return enriched

        symmetric = enrich_rows(metrics.get("symmetric", []))
        report = {
            "default_num": int(default_num),
            "protocol": metrics.get("protocol", {}),
            "macro": metrics.get("macro", {}),
            "logged_macro": result,
            "symmetric": symmetric,
        }
        with open(os.path.join(infer_dir, "any_modal_metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        self._write_inference_csv(os.path.join(infer_dir, "symmetric_results.csv"), symmetric, ranks)

        vis_cfg = self.cfg.get("inference", {}).get("visualize", {})
        if vis_cfg.get("enabled", True):
            per_pair = {f'{row["query"]}->{row["gallery"]}': row for row in symmetric}
            self._plot_inference_report(per_pair, os.path.join(infer_dir, "any_modal_metrics.png"))

    def _write_inference_csv(self, path: str, rows, ranks) -> None:
        import csv

        fieldnames = [
            "query",
            "gallery",
            "query_modalities",
            "gallery_modalities",
            "query_missing",
            "gallery_missing",
            "query_available_count",
            "gallery_available_count",
            "mAP",
            *[f"Rank{rank}" for rank in ranks],
            "source",
        ]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                item = dict(row)
                for key in ("query_modalities", "gallery_modalities", "query_missing", "gallery_missing"):
                    item[key] = "+".join(str(v) for v in item.get(key, []))
                writer.writerow({key: item.get(key, "") for key in fieldnames})

    def _plot_inference_report(self, per_pair: Dict[str, Dict[str, float]], path: str) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")  # file-only rendering; never depend on an interactive GUI backend
            import matplotlib.pyplot as plt
        except ImportError:
            return
        if not per_pair:
            return
        names = list(per_pair.keys())
        maps = [per_pair[name]["mAP"] for name in names]
        plt.figure(figsize=(max(6.0, 0.8 * len(names)), 4.0))
        plt.bar(names, maps)
        plt.ylabel("mAP")
        plt.ylim(0.0, 1.0)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()

    def train(self, train_loader, query_loader=None, gallery_loader=None, train_eval_loader=None) -> None:
        init_tracker(self.cfg, self.cfg)
        self._archive_run_state()
        self._log_model_summary()
        if self.resume_requested:
            self._resume_load_payload()
        self._initialize_checkpoint_files()
        per_gpu_batch = int(getattr(train_loader, "batch_size", 1) or 1)
        global_batch = per_gpu_batch * self.world_size
        updates_per_epoch = len(train_loader)
        total_iters = len(train_loader) * self.max_epochs
        total_updates = updates_per_epoch * self.max_epochs
        self.scheduler = build_lr_scheduler(self.cfg, self.optimizer, total_steps=total_updates)
        self._resume_restore_schedule()
        if self.rank == 0 and self.resume_requested:
            if self.resume_memory:
                note = (
                    f"Breakpoint training: resumed from {self.resume_path}; "
                    f"continuing at epoch {self.epochs_completed + 1}/{self.max_epochs}, "
                    f"global update {self.global_update}, "
                    f"best epoch-average train loss {self.best_train_loss:.4f}"
                    + (f"@epoch{self.best_epoch}" if self.best_epoch > 0 else " (pending)")
                    + ". Backbone pretrained / pretrained_weights_path are ignored; "
                    "all other settings follow the YAML config; logs are appended."
                )
                event = "breakpoint_resume"
            else:
                note = (
                    f"Breakpoint no-memory training: loaded model weights from {self.resume_path}; "
                    f"starting at epoch 1/{self.max_epochs}, global update 0. "
                    "All training state is reset and all settings follow the current YAML; "
                    "backbone pretrained / pretrained_weights_path are ignored."
                )
                event = "breakpoint_no_memory"
            print(note)
            log_event(
                event,
                step=self.global_update,
                checkpoint=str(self.resume_path),
                checkpoint_name=os.path.basename(str(self.resume_path)),
                epochs_completed=int(self.epochs_completed),
                resume_epoch=int(self.epochs_completed + 1),
                global_update=int(self.global_update),
                best_train_loss=(
                    float(self.best_train_loss) if math.isfinite(self.best_train_loss) else None
                ),
                best_epoch=int(self.best_epoch),
                has_trainer_state=self._resume_state is not None,
                memory_restored=bool(self.resume_memory),
            )
            log_scalars(
                {
                    "train/resumed": 1.0,
                    "train/resume_memory": float(self.resume_memory),
                    "train/resume_epochs_completed": float(self.epochs_completed),
                    "train/resume_global_update": float(self.global_update),
                },
                self.global_update,
            )
        if self.rank == 0:
            check_interval = self.early_stop_cfg.get("check_interval_iters", "auto")
            if isinstance(check_interval, str) and check_interval.lower() == "auto":
                self.early_stop_check_interval_iters = min(max(1, updates_per_epoch), 500)
            else:
                self.early_stop_check_interval_iters = max(1, int(check_interval))
            print(
                f"Training plan: {self.max_epochs} epochs, "
                f"{len(train_loader)} iterations per epoch, "
                f"{updates_per_epoch} optimizer updates per epoch, "
                f"{total_iters} total iterations, {total_updates} total updates, "
                f"global batch size {global_batch}, "
                f"warmup {self.warmup_updates} successful optimizer updates "
                f"best checkpoint starts after warmup update {self.warmup_updates}, "
                f"early-stop checks every {self.early_stop_check_interval_iters} updates "
                f"with patience {self.early_stop_patience_checks} checks."
            )
            if train_eval_loader is not None:
                print(
                    f"Train accuracy evaluation: every {self.train_eval_period_epochs} epoch(s), "
                    f"{len(train_eval_loader.dataset)} training images, batch size {train_eval_loader.batch_size}."
                )
            else:
                print("Train accuracy evaluation: disabled.")
        if self.is_ddp:
            interval_t = torch.tensor([int(self.early_stop_check_interval_iters)], device=self.device)
            patience_t = torch.tensor([int(self.early_stop_patience_checks)], device=self.device)
            dist.broadcast(interval_t, src=0)
            dist.broadcast(patience_t, src=0)
            self.early_stop_check_interval_iters = int(interval_t.item())
            self.early_stop_patience_checks = int(patience_t.item())

        epoch = self.epochs_completed
        if self.rank == 0 and self.resume_requested and self.resume_memory and epoch >= self.max_epochs:
            print(
                f"Breakpoint training: checkpoint already reached {epoch}/{self.max_epochs} "
                "epochs; no further training is needed. Running final evaluation only."
            )
        while epoch < self.max_epochs:
            stats = self.train_epoch(train_loader, epoch)
            epoch += 1
            self.epochs_completed = epoch
            stop_now = bool(stats["stop"])
            if self.rank == 0:
                best_active = self.global_update > self.warmup_updates
                improved = (
                    self._save_best_by_train_loss(stats["loss"], self.global_update, epoch)
                    if best_active
                    else False
                )
                epoch_log = {"epoch": epoch, "epoch/loss": stats["loss"]}
                epoch_log.update(
                    {
                        "epoch/time_sec": stats["time_sec"],
                        "epoch/time_per_iter_sec": stats["time_per_iter_sec"],
                        "epoch/time_per_update_sec": stats["time_per_update_sec"],
                        "epoch/samples_per_sec": stats["samples_per_sec"],
                    }
                )
                if self.best_epoch > 0:
                    epoch_log["train/best_epoch_avg_loss"] = self.best_train_loss
                log_scalars(epoch_log, self.global_update)
                best_text = (
                    f"{self.best_train_loss:.4f}@epoch{self.best_epoch}"
                    if self.best_epoch > 0
                    else "pending"
                )
                suffix = " improved" if improved else ""
                print(
                    f"Epoch {epoch}/{self.max_epochs} done "
                    f"epoch_avg_loss={stats['loss']:.4f} "
                    f"time={stats['time_sec']:.2f}s "
                    f"time/iter={stats['time_per_iter_sec']:.3f}s "
                    f"time/update={stats['time_per_update_sec']:.3f}s "
                    f"throughput={stats['samples_per_sec']:.1f} samples/s "
                    f"best_epoch_avg={best_text}{suffix}"
                )
                self._save_latest(self.global_update)
            if self._should_evaluate_train_accuracy(train_eval_loader, epoch):
                self.evaluate_train_accuracy(train_eval_loader, epoch)
            if stop_now:
                if self.rank == 0:
                    print(
                        f"Early stop at global update {self.global_update}; "
                        f"best epoch-average train loss was at epoch {self.best_epoch} "
                        f"(update {self.best_iter}); early-stop window did not improve for "
                        f"{self.bad_stop_checks} checks."
                )
                break

        if query_loader is not None and gallery_loader is not None:
            self.test(query_loader, gallery_loader)
        tracker_finish()
