"""MV-SRBF: shared backbone per modality, cross-modal recovery, fusion."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.mv_srbf_losses import (
    ortho_intra_loss,
    ortho_inter_loss,
)
from losses.reid_losses import (
    CenterLoss,
    classification_loss,
    cross_modal_triplet_loss,
    metric_learning_loss,
)
from models.backbones.registry import build_backbone
from models.reid_heads import ModalReIDHead
from utils.weight_init import init_kaiming


def resolve_modal_keys(dataset_cfg: Dict[str, Any]) -> List[str]:
    """Resolve modality keys from dataset config and fall back to num_modalities."""
    modalities = dataset_cfg.get("modalities", {})
    if modalities:
        return list(modalities.keys())
    num_modalities = int(dataset_cfg.get("num_modalities", 1))
    return [f"modal{i + 1}" for i in range(num_modalities)]


class CrossModalRecovery(nn.Module):
    """Stacked cross-modal MHA recovery blocks over modality CLS tokens."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 2.0, num_layers: int = 1):
        super().__init__()
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError("recovery.num_layers must be positive.")
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.norm = nn.LayerNorm(dim)
        self.additional_layers = nn.ModuleList(
            [CrossModalRecovery(dim, num_heads, mlp_ratio, num_layers=1) for _ in range(self.num_layers - 1)]
        )
        self.mlp.apply(init_kaiming)
        self.norm.apply(init_kaiming)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(self.mlp(attn_out + x))
        for layer in self.additional_layers:
            x = layer(x)
        return x


class SharedModalHeadDict(nn.Module):
    """Expose one shared modal head through the same key-based API as ModuleDict."""

    def __init__(self, modal_keys: List[str], head: nn.Module):
        super().__init__()
        self.modal_keys = list(modal_keys)
        self.shared = head

    def __getitem__(self, key: str) -> nn.Module:
        if key not in self.modal_keys:
            raise KeyError(key)
        return self.shared

    def keys(self):
        return self.modal_keys

    def items(self):
        for key in self.modal_keys:
            yield key, self.shared


class LocalGlobalReducer(nn.Module):
    """Fuse ViT CLS and pooled patch tokens while preserving descriptor width."""

    def __init__(self, dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
        )
        self.layers.apply(init_kaiming)

    def forward(self, cls_token: torch.Tensor, spatial_tokens: torch.Tensor) -> torch.Tensor:
        local = spatial_tokens.mean(dim=1)
        return self.layers(torch.cat([cls_token, local], dim=-1))


class MVSRBF(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        ds = cfg["dataset"]
        self.modal_keys = resolve_modal_keys(ds)
        self.num_modalities = len(self.modal_keys)
        self.num_classes = int(ds.get("num_classes", cfg["model"]["heads"].get("num_classes", 100)))

        self.encoder, self.family = build_backbone(cfg)
        head_cfg = cfg["model"].get("heads", {})
        emb = int(getattr(self.encoder, "output_dim", 0))
        if emb <= 0:
            raise ValueError(
                "Cannot infer model feature dimension from the backbone. "
                "Use a supported backbone adapter; custom adapters must expose an output_dim attribute."
            )
        self.feature_dim = emb
        mcfg = cfg["model"]["mv_srbf"]
        n_heads = int(mcfg.get("n_recovery_heads", 8))
        rec_cfg = mcfg.get("recovery", {})
        recovery_num_layers = int(rec_cfg.get("num_layers", 1))
        self.recovery_enabled = bool(rec_cfg.get("enabled", True))
        if self.recovery_enabled:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, emb))
            nn.init.normal_(self.mask_token, std=0.02)
            self.recovery = CrossModalRecovery(emb, num_heads=n_heads, num_layers=recovery_num_layers)
        fusion_cfg = mcfg.get("fusion", {})
        if "modal_mlp" in fusion_cfg:
            raise ValueError("model.mv_srbf.fusion.modal_mlp is internal and not configurable.")
        self.modal_feature_dim = emb
        self.modal_mlp = nn.Sequential(
            nn.Linear(emb, emb),
            nn.GELU(),
            nn.Linear(emb, emb),
        )
        self.modal_mlp.apply(init_kaiming)
        fuse_in_dim = self.modal_feature_dim * self.num_modalities
        self.fusion_mode = str(fusion_cfg.get("mode", "single")).strip().lower()
        if self.fusion_mode not in {"single", "full", "concat"}:
            raise ValueError(
                f"Unknown model.mv_srbf.fusion.mode {self.fusion_mode!r}. Use 'single' "
                "(fuse to one modal width), 'full' (MLP over the full concatenated "
                "modality width), or 'concat' (concatenate without a fusion MLP)."
            )
        # single -> one modal width; full/concat -> full concatenated width.
        self.fuse_out_dim = self.modal_feature_dim if self.fusion_mode == "single" else fuse_in_dim
        local_global_cfg = head_cfg.get("local_global", {})
        self.local_global_enabled = bool(local_global_cfg.get("enabled", False)) and self.family == "vit"
        self.local_global_reducers = nn.ModuleDict()
        if self.local_global_enabled:
            self.local_global_reducers = nn.ModuleDict({key: LocalGlobalReducer(emb) for key in self.modal_keys})
        with_normneck = bool(head_cfg.get("with_normneck", True))
        if "norm" + "_type" in head_cfg:
            raise ValueError("Custom head neck normalization is not supported.")
        head_sharing = head_cfg.get("sharing", {})
        self.share_modal_heads = bool(head_sharing.get("modal_heads", False))
        self.share_modal_centers = bool(head_sharing.get("modal_centers", False))
        if self.fusion_mode == "concat":
            # Directly concatenate modality features; no fusion MLP.
            self.fuse_mlp = nn.Identity()
        else:
            # single -> reduce to one modal width; full -> keep the concatenated width.
            self.fuse_mlp = nn.Sequential(
                nn.Linear(fuse_in_dim, fuse_in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(fuse_in_dim, self.fuse_out_dim),
            )
            self.fuse_mlp.apply(init_kaiming)
        self.fuse_head = ModalReIDHead(
            self.family,
            self.fuse_out_dim,
            self.num_classes,
            with_neck=with_normneck,
        )

        if self.share_modal_heads:
            self.modal_heads = SharedModalHeadDict(
                self.modal_keys,
                ModalReIDHead(
                    self.family,
                    self.modal_feature_dim,
                    self.num_classes,
                    with_neck=with_normneck,
                ),
            )
        else:
            self.modal_heads = nn.ModuleDict(
                {
                    k: ModalReIDHead(
                        self.family,
                        self.modal_feature_dim,
                        self.num_classes,
                        with_neck=with_normneck,
                    )
                    for k in self.modal_keys
                }
            )
        if self.share_modal_centers:
            self.center_modal = nn.ModuleList([CenterLoss(self.num_classes, self.modal_feature_dim)])
        else:
            self.center_modal = nn.ModuleList([CenterLoss(self.num_classes, self.modal_feature_dim) for _ in self.modal_keys])
        self.center_fuse = CenterLoss(self.num_classes, self.fuse_out_dim)
        self.center_cross = CenterLoss(self.num_classes, emb)

    def _modal_center(self, index: int) -> CenterLoss:
        return self.center_modal[0] if self.share_modal_centers else self.center_modal[index]

    def encode_modalities(self, modalities: Dict[str, torch.Tensor], aux_indices: Optional[Dict[int, torch.Tensor]] = None):
        cls_list = []
        spat_list = []
        spatial_hw_list = []
        for k in self.modal_keys:
            if k not in modalities:
                raise KeyError(f"Missing modality {k!r}; available keys: {sorted(modalities.keys())}")
            batch_size = modalities[k].shape[0]
            modality_id = self.modal_keys.index(k)
            modality_ids = torch.full((batch_size,), modality_id, dtype=torch.long, device=modalities[k].device)
            out = self.encoder(modalities[k], modality_ids=modality_ids, aux_indices=aux_indices)
            cls_pre = out.cls_pre
            if self.local_global_enabled:
                if out.spatial_tokens is None:
                    raise ValueError("heads.local_global.enabled requires ViT spatial tokens.")
                cls_pre = self.local_global_reducers[k](cls_pre, out.spatial_tokens)
            cls_list.append(cls_pre)
            spat_list.append(out.spatial_tokens)
            spatial_hw_list.append(out.spatial_hw)
        cls_stack = torch.stack(cls_list, dim=1)
        return cls_stack, spat_list, spatial_hw_list

    def _modal_head_outputs(self, tokens: torch.Tensor):
        """Apply private per-modality norm necks/classifiers to [B, M, D] tokens."""
        logits = {}
        after_list = []
        pre_list = []
        for i, key in enumerate(self.modal_keys):
            logit, feat_after, feat_pre = self.modal_heads[key](tokens[:, i])
            logits[key] = logit
            after_list.append(feat_after)
            pre_list.append(feat_pre)
        return logits, torch.stack(after_list, dim=1), torch.stack(pre_list, dim=1)

    def classify_full_modalities(
        self,
        modalities: Dict[str, torch.Tensor],
        aux_indices: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[str, Dict[str, torch.Tensor] | torch.Tensor]:
        """Return per-modality and full-fusion logits for deterministic train-set accuracy."""
        cls_stack, _, _ = self.encode_modalities(modalities, aux_indices=aux_indices)
        modal_logits = {}
        modal_tokens_pre_neck = self.modal_mlp(cls_stack)
        modal_logits, modal_tokens_after_neck, _ = self._modal_head_outputs(modal_tokens_pre_neck)
        fuse_pre = self.fuse_mlp(modal_tokens_after_neck.flatten(1))
        fuse_logits, _, _ = self.fuse_head(fuse_pre)
        return {"modal_logits": modal_logits, "fuse_logits": fuse_logits}

    def fuse_inference_features(
        self,
        modal_features: torch.Tensor,
        *,
        neck_feat: str = "before",
    ) -> torch.Tensor:
        """Apply the trained fusion path to completed per-modality features."""
        fuse_pre = self.fuse_mlp(modal_features.flatten(1))
        _, fuse_after, fuse_before = self.fuse_head(fuse_pre)
        return fuse_after if str(neck_feat).lower() == "after" else fuse_before

    def sample_mask(self, b: int, m: int, device: torch.device) -> torch.Tensor:
        """
        Sample masked-modality sets with 0 <= N_mask < M.

        True means the modality CLS token is replaced by the learnable mask token.
        The all-visible case is valid; recovery is bypassed for that sample.
        """
        if not self.recovery_enabled:
            return torch.zeros(b, max(0, m), dtype=torch.bool, device=device)
        if m <= 1:
            return torch.zeros(b, max(0, m), dtype=torch.bool, device=device)
        valid: List[torch.Tensor] = []
        for bits in range(1 << m):
            row = torch.tensor([(bits >> j) & 1 for j in range(m)], dtype=torch.bool, device=device)
            if row.all():
                continue
            valid.append(row)
        pick = torch.randint(0, len(valid), (b,), device=device)
        return torch.stack([valid[int(pick[i])] for i in range(b)], dim=0)

    def _validate_cls_mask(self, mask: torch.Tensor, batch_size: int, num_modalities: int) -> torch.Tensor:
        """Validate paper constraint N_mask < |M| for each sample."""
        if mask.shape != (batch_size, num_modalities):
            raise ValueError(f"Mask shape must be {(batch_size, num_modalities)}, got {tuple(mask.shape)}.")
        mask = mask.to(dtype=torch.bool)
        if num_modalities > 1 and mask.all(dim=1).any():
            raise ValueError("Invalid modality mask: N_mask must be smaller than the number of modalities.")
        if num_modalities == 1 and mask.any():
            raise ValueError("Single-modality input cannot be masked because no reference modality would remain.")
        return mask

    def _zero_recovery_losses(self, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = reference.new_zeros(())
        return {
            "rec_cls_mse": zero,
            "rec": zero,
        }

    def forward(
        self,
        modalities: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        aux_indices: Optional[Dict[int, torch.Tensor]] = None,
        return_loss_dict: bool = True,
        use_recovery: Optional[bool] = None,
        use_fusion: Optional[bool] = None,
        loss_options: Optional[Dict[str, bool]] = None,
    ):
        loss_options = loss_options or {}
        cls_stack, spat_list, _ = self.encode_modalities(modalities, aux_indices=aux_indices)
        b, m, d = cls_stack.shape
        device = cls_stack.device
        recovery_active = self.recovery_enabled if use_recovery is None else self.recovery_enabled and bool(use_recovery)
        fusion_active = True if use_fusion is None else bool(use_fusion)
        compute_encoding = bool(loss_options.get("encoding", True))
        compute_fusion_loss = bool(loss_options.get("fusion", True))
        compute_cross = bool(loss_options.get("cross_modal", True))
        compute_enc_ce = bool(loss_options.get("enc_ce", compute_encoding))
        compute_enc_tri = bool(loss_options.get("enc_tri", compute_encoding))
        compute_enc_center = bool(loss_options.get("enc_center", compute_encoding))
        compute_fuse_ce = bool(loss_options.get("fuse_ce", compute_fusion_loss))
        compute_fuse_tri = bool(loss_options.get("fuse_tri", compute_fusion_loss))
        compute_fuse_center = bool(loss_options.get("fuse_center", compute_fusion_loss))
        compute_cross_tri = bool(loss_options.get("cross_tri", compute_cross))
        compute_cross_center = bool(loss_options.get("cross_center", compute_cross))
        compute_ortho_intra = bool(loss_options.get("ortho_intra", True))
        compute_ortho_inter = bool(loss_options.get("ortho_inter", True))

        if mask is None and self.training and recovery_active:
            mask = self.sample_mask(b, m, device)
        elif mask is not None:
            mask = self._validate_cls_mask(mask, b, m).to(device=device)

        modal_tokens_pre_neck = self.modal_mlp(cls_stack)
        if not self.training or compute_enc_ce or compute_enc_tri or compute_enc_center:
            modal_logits, modal_tokens_after_neck, modal_tokens_metric = self._modal_head_outputs(
                modal_tokens_pre_neck
            )
        else:
            modal_logits = {}
            modal_tokens_after_neck = modal_tokens_pre_neck
            modal_tokens_metric = modal_tokens_pre_neck
        teacher = cls_stack.detach()

        recovered = None
        if recovery_active:
            cls_in = cls_stack.clone()
            if mask is not None and mask.any():
                mt = self.mask_token.expand(b, m, d)
                cls_in = torch.where(mask.unsqueeze(-1), mt, cls_in)
            recovered = self.recovery(cls_in)
            fuse_tokens_pre_mlp = torch.where(mask.unsqueeze(-1), recovered, cls_stack) if mask is not None else cls_stack
        else:
            if mask is not None and mask.any():
                fuse_tokens_pre_mlp = cls_stack.masked_fill(mask.unsqueeze(-1), 0.0)
            else:
                fuse_tokens_pre_mlp = cls_stack
        # Fusion order: noise -> per-modality MLP -> orthogonality -> final fusion MLP.
        fuse_tokens_mlp_input = fuse_tokens_pre_mlp
        if fusion_active and self.training:
            noise_ratio = float(self.cfg["model"]["mv_srbf"].get("fusion_noise_ratio", 0.1))
            if noise_ratio > 0:
                sigma = fuse_tokens_pre_mlp.std() * noise_ratio
                fuse_tokens_mlp_input = fuse_tokens_pre_mlp + sigma * torch.randn_like(fuse_tokens_pre_mlp)
        fuse_tokens_pre_neck = self.modal_mlp(fuse_tokens_mlp_input)
        _, fuse_tokens_after_neck, _ = self._modal_head_outputs(fuse_tokens_pre_neck)

        o_intra = cls_stack.new_zeros(())
        if compute_ortho_intra:
            for st in spat_list:
                if st is not None:
                    o_intra = o_intra + ortho_intra_loss(st)
        o_inter = ortho_inter_loss(fuse_tokens_after_neck) if compute_ortho_inter else cls_stack.new_zeros(())

        if fusion_active:
            fuse_pre = self.fuse_mlp(fuse_tokens_after_neck.flatten(1))
            fuse_logits, fuse_id, fuse_pre_metric = self.fuse_head(fuse_pre)
        else:
            num_classes = int(self.cfg["model"]["heads"].get("num_classes", self.cfg["dataset"]["num_classes"]))
            fuse_logits = cls_stack.new_zeros((b, num_classes))
            fuse_id = cls_stack.new_zeros((b, self.fuse_out_dim))
            fuse_pre_metric = cls_stack.new_zeros((b, self.fuse_out_dim))

        loss_dict: Dict[str, torch.Tensor] = {}
        if return_loss_dict and self.training:
            loss_cfg = self.cfg["model"]["losses"]
            triplet_cfg = loss_cfg["triplet"]
            ce_cfg = loss_cfg["ce"]

            if not recovery_active:
                loss_dict.update(self._zero_recovery_losses(cls_stack))
            elif mask is not None and mask.any():
                rec_cfg = self.cfg["model"]["mv_srbf"].get("recovery", {})
                cls_mse = F.mse_loss(recovered[mask], teacher[mask])
                loss_dict["rec_cls_mse"] = cls_mse
                loss_dict["rec"] = float(rec_cfg.get("scale", 1.0)) * cls_mse
            else:
                loss_dict["rec_cls_mse"] = cls_stack.new_zeros(())
                loss_dict["rec"] = cls_stack.new_zeros(())

            loss_dict["ortho_intra"] = o_intra
            loss_dict["ortho_inter"] = o_inter

            enc_ce = cls_stack.new_zeros(())
            enc_tri = cls_stack.new_zeros(())
            enc_center = cls_stack.new_zeros(())
            if compute_enc_ce or compute_enc_tri or compute_enc_center:
                for i, k in enumerate(self.modal_keys):
                    head = self.modal_heads[k]
                    logits = modal_logits[k]
                    feat_id = modal_tokens_after_neck[:, i]
                    pre = modal_tokens_metric[:, i]
                    if compute_enc_ce:
                        enc_ce = enc_ce + classification_loss(
                            logits,
                            labels,
                            ce_cfg,
                            features=feat_id,
                            classifier_weight=head.classifier.weight,
                        )
                    if compute_enc_tri:
                        enc_tri = enc_tri + metric_learning_loss(pre, labels, triplet_cfg)
                    if compute_enc_center:
                        enc_center = enc_center + self._modal_center(i)(pre, labels)

            fuse_ce = cls_stack.new_zeros(())
            fuse_tri = cls_stack.new_zeros(())
            fuse_center = cls_stack.new_zeros(())
            if fusion_active:
                if compute_fuse_ce:
                    fuse_ce = classification_loss(
                        fuse_logits,
                        labels,
                        ce_cfg,
                        features=fuse_id,
                        classifier_weight=self.fuse_head.classifier.weight,
                    )
                if compute_fuse_tri:
                    fuse_tri = metric_learning_loss(fuse_pre_metric, labels, triplet_cfg)
                if compute_fuse_center:
                    fuse_center = self.center_fuse(fuse_pre_metric, labels)
            cross_cfg = loss_cfg.get("cross_modal", {})
            cross_triplet_cfg = cross_cfg.get("triplet", {})
            cross_center_cfg = cross_cfg.get("center", {})
            cross_tri = (
                cross_modal_triplet_loss(
                    cls_stack,
                    labels,
                    cross_triplet_cfg,
                )
                if compute_cross_tri and bool(cross_triplet_cfg.get("enabled", False))
                else cls_stack.new_zeros(())
            )
            cross_center = (
                self.center_cross(cls_stack.reshape(b * m, d), labels.view(b, 1).expand(b, m).reshape(-1))
                if compute_cross_center and bool(cross_center_cfg.get("enabled", False))
                else cls_stack.new_zeros(())
            )

            loss_dict["enc_ce"] = enc_ce
            loss_dict["enc_tri"] = enc_tri
            loss_dict["enc_center"] = enc_center
            loss_dict["fuse_ce"] = fuse_ce
            loss_dict["fuse_tri"] = fuse_tri
            loss_dict["fuse_center"] = fuse_center
            loss_dict["cross_tri"] = cross_tri
            loss_dict["cross_center"] = cross_center

        return {
            "fuse_logits": fuse_logits,
            "fuse_feat_pre": fuse_pre_metric,
            "fuse_feat_after": fuse_id,
            "cls_stack": cls_stack,
            "stage_features": {
                "modal": {
                    "before": modal_tokens_metric,
                    "after": modal_tokens_after_neck,
                    "fusion_input": modal_tokens_after_neck,
                },
                "recover": {
                    "before": fuse_tokens_pre_neck,
                    "after": fuse_tokens_after_neck,
                    "fusion_input": fuse_tokens_after_neck,
                },
            },
            "feature_ortho_intra": fuse_tokens_pre_mlp.flatten(1),
            "feature_ortho_inter": fuse_tokens_pre_neck.flatten(1),
            "feature_ortho_inter_before": fuse_tokens_pre_neck.flatten(1),
            "feature_ortho_inter_after": fuse_tokens_after_neck.flatten(1),
            "modal_keys": self.modal_keys,
            "loss_dict": loss_dict if return_loss_dict else {},
        }
