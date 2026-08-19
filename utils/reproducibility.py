"""Reproducibility helpers shared by training and inference entry points."""
from __future__ import annotations

import os
import random
from functools import partial
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and torch with the same base seed on every rank."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    return seed


def seed_worker(worker_id: int, base_seed: int, rank: int = 0) -> None:
    """Seed a DataLoader worker deterministically and differently per DDP rank."""
    worker_seed = int(base_seed) + int(rank) * 100000 + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)


def build_worker_init_fn(base_seed: int, rank: int = 0):
    """Return a worker_init_fn closure for torch DataLoader."""
    return partial(seed_worker, base_seed=base_seed, rank=rank)


def build_torch_generator(base_seed: int, rank: int = 0) -> Optional[torch.Generator]:
    """Create a rank-aware torch Generator for DataLoader internals."""
    generator = torch.Generator()
    generator.manual_seed(int(base_seed) + int(rank))
    return generator
