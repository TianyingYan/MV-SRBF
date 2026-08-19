"""
Build train / query / gallery DataLoaders using get_dataset + OK samplers.
"""
from __future__ import annotations

from typing import Any, Dict

from torch.utils.data import DataLoader
import torch.distributed as dist

from data.datasets.base import get_dataset
from data.samplers import build_training_sampler, build_query_sampler
from data.samplers.inference_sampler import InferenceSampler
from data.transforms.multi_spectral_transforms import MultiSpectralTransform
from data.utils.collate import collate_multispectral_batch
from utils.config import resolve_runtime_path
from utils.input_config import resolve_input_padding, resolve_input_size, resolve_resize_interpolation
from utils.reproducibility import build_torch_generator, build_worker_init_fn

__all__ = ["build_dataloaders", "build_multispectral_transform_from_cfg"]


def _merge_dataset_metadata(cfg: Dict[str, Any], *datasets) -> None:
    metadata: Dict[str, Any] = {}
    aux_values: Dict[int, set] = {}
    for ds in datasets:
        ds_meta = getattr(ds, "metadata", {}) or {}
        for dim, meta in ds_meta.get("aux_dims", {}).items():
            values = meta.get("values", []) if isinstance(meta, dict) else []
            aux_values.setdefault(int(dim), set()).update(int(v) for v in values)
        if not metadata:
            metadata.update(ds_meta)
            continue
        metadata["num_cameras"] = max(int(metadata.get("num_cameras", 0)), int(ds_meta.get("num_cameras", 0)))
        metadata["has_camera"] = bool(metadata.get("has_camera", False) or ds_meta.get("has_camera", False))
        if not metadata.get("modal_keys") and ds_meta.get("modal_keys"):
            metadata["modal_keys"] = ds_meta["modal_keys"]
            metadata["num_modalities"] = ds_meta.get("num_modalities", len(ds_meta["modal_keys"]))
    if metadata:
        merged_aux = {}
        for dim, values in aux_values.items():
            ordered = sorted(values)
            merged_aux[int(dim)] = {
                "values": ordered,
                "num_values": len(ordered),
                "value_to_index": {int(v): i for i, v in enumerate(ordered)},
            }
        metadata["aux_dims"] = merged_aux
        train_num_classes = int((getattr(datasets[0], "metadata", {}) or {}).get("num_classes", 0))
        if train_num_classes > 0:
            metadata["num_classes"] = train_num_classes
            cfg.setdefault("dataset", {})["num_classes"] = train_num_classes
        cam_dim = int(metadata.get("camera_aux_dim", 0))
        if cam_dim in merged_aux:
            metadata["num_cameras"] = int(merged_aux[cam_dim]["num_values"])
            metadata["has_camera"] = metadata["num_cameras"] > 0
        for ds in datasets:
            ds.metadata = dict(metadata)
            ds.aux_value_to_index = {
                int(dim): {int(k): int(v) for k, v in meta.get("value_to_index", {}).items()}
                for dim, meta in merged_aux.items()
            }
        cfg.setdefault("dataset", {})["metadata"] = metadata


def resolve_dataset_root(root: str, path_base: str | None = None) -> str:
    """Resolve a dataset root from the runtime path base captured by the config."""
    return resolve_runtime_path(root, path_base)


def build_multispectral_transform_from_cfg(cfg: Dict[str, Any], split: str = "train") -> MultiSpectralTransform:
    inp = cfg["input"]
    h, w = resolve_input_size(cfg)
    aug_key = "train_augmentation" if split == "train" else "inference_augmentation"
    aug = inp[aug_key]
    if "per" + "_modal" in aug:
        raise ValueError("Unsupported modality-specific augmentation block.")
    normalize_params = inp.get("normalize", {})
    padding = resolve_input_padding(cfg)
    return MultiSpectralTransform(
        input_size=(h, w),
        interpolation=resolve_resize_interpolation(cfg),
        normalize_params=normalize_params,
        shared_aug=aug.get("shared", {}),
        padding=padding,
        padding_fill=inp.get("padding_fill", 0),
    )


def build_dataloaders(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construct dataloaders from merged experiment config.

    Train batch size is O*K per process; DDP uses world_size*O*K samples per optimizer step.
    """
    ds_cfg = cfg["dataset"]
    name = ds_cfg["name"]
    root = resolve_dataset_root(ds_cfg["root"], cfg.get("_path_base"))
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    seed = int(cfg.get("seed", 42))
    worker_init_fn = build_worker_init_fn(seed, rank)
    generator = build_torch_generator(seed, rank)

    train_tf = build_multispectral_transform_from_cfg(cfg, "train")
    query_tf = build_multispectral_transform_from_cfg(cfg, "inference")
    gallery_tf = build_multispectral_transform_from_cfg(cfg, "inference")

    train_ds = get_dataset(name, root=root, split="train", transform=train_tf, verbose=False)
    train_eval_ds = get_dataset(name, root=root, split="train", transform=query_tf, verbose=False)
    query_ds = get_dataset(name, root=root, split="query", transform=query_tf, verbose=False)
    gallery_ds = get_dataset(name, root=root, split="gallery", transform=gallery_tf, verbose=False)
    _merge_dataset_metadata(cfg, train_ds, train_eval_ds, query_ds, gallery_ds)

    train_sampler = build_training_sampler(cfg, train_ds)

    dl_train_cfg = cfg["dataloader"]["train"]
    o = dl_train_cfg["sampler"]["o_ids_per_gpu"]
    k = dl_train_cfg["sampler"]["k_instances_per_id"]
    per_gpu_batch = o * k

    train_loader = DataLoader(
        train_ds,
        batch_size=per_gpu_batch,
        sampler=train_sampler,
        num_workers=int(dl_train_cfg.get("num_workers", 4)),
        pin_memory=bool(dl_train_cfg.get("pin_memory", True)),
        drop_last=bool(dl_train_cfg.get("drop_last", True)),
        collate_fn=collate_multispectral_batch,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    q_cfg = cfg["dataloader"]["query"]
    g_cfg = cfg["dataloader"].get("gallery", q_cfg)
    q_bs = int(q_cfg.get("batch_size", 32))
    g_bs = int(g_cfg.get("batch_size", q_bs))

    query_sampler = build_query_sampler(cfg, query_ds)
    # Any-modal evaluation runs only on rank 0 without cross-rank feature gather, so
    # the gallery must be covered in full on rank 0; pin to a single replica instead
    # of letting InferenceSampler shard the gallery per DDP rank.
    if "gallery" in cfg["dataloader"]:
        g_s = cfg["dataloader"]["gallery"]["sampler"]
        gallery_sampler = InferenceSampler(
            gallery_ds,
            num_replicas=1,
            rank=0,
            shuffle=g_s["shuffle"],
            seed=cfg["seed"],
            drop_last=g_s["drop_last"],
        )
    else:
        gallery_sampler = InferenceSampler(
            gallery_ds,
            num_replicas=1,
            rank=0,
            shuffle=q_cfg["sampler"]["shuffle"],
            seed=cfg["seed"],
            drop_last=q_cfg["sampler"]["drop_last"],
        )

    query_loader = DataLoader(
        query_ds,
        batch_size=q_bs,
        sampler=query_sampler,
        num_workers=int(q_cfg.get("num_workers", 2)),
        pin_memory=bool(q_cfg.get("pin_memory", False)),
        drop_last=False,
        collate_fn=collate_multispectral_batch,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    gallery_loader = DataLoader(
        gallery_ds,
        batch_size=g_bs,
        sampler=gallery_sampler,
        num_workers=int(g_cfg.get("num_workers", 2)),
        pin_memory=bool(g_cfg.get("pin_memory", False)),
        drop_last=False,
        collate_fn=collate_multispectral_batch,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    train_eval_loader = None
    train_eval_cfg = cfg["dataloader"].get("train_eval", {})
    if bool(train_eval_cfg.get("enabled", True)):
        te_sampler_cfg = train_eval_cfg.get("sampler", {})
        # Train accuracy should cover every training image once on rank 0.
        train_eval_sampler = InferenceSampler(
            train_eval_ds,
            num_replicas=1,
            rank=0,
            shuffle=bool(te_sampler_cfg.get("shuffle", False)),
            seed=cfg["seed"],
            drop_last=bool(te_sampler_cfg.get("drop_last", False)),
        )
        train_eval_loader = DataLoader(
            train_eval_ds,
            batch_size=int(train_eval_cfg.get("batch_size", q_bs)),
            sampler=train_eval_sampler,
            num_workers=int(train_eval_cfg.get("num_workers", q_cfg.get("num_workers", 2))),
            pin_memory=bool(train_eval_cfg.get("pin_memory", q_cfg.get("pin_memory", False))),
            drop_last=False,
            collate_fn=collate_multispectral_batch,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )

    return {
        "train_loader": train_loader,
        "train_eval_loader": train_eval_loader,
        "query_loader": query_loader,
        "gallery_loader": gallery_loader,
        "train_sampler": train_sampler,
        "num_train_samples": len(train_ds),
    }
