import math
import numpy as np
import torch
from torch.utils.data.sampler import Sampler

class InferenceSampler(Sampler):
    """
    Sequential, deterministic sampler for inference.
    Works in CPU, single-GPU, and DistributedDataParallel (DDP) inference.

    - In DDP: each rank gets a disjoint contiguous slice.
    - In DP / single-GPU: returns all indices.
    - Always returns indices in order (no shuffle).
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0, drop_last=False):
        self.dataset = dataset

        # Detect DDP environment
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            self.num_replicas = 1
            self.rank = 0
        else:
            self.num_replicas = torch.distributed.get_world_size() if num_replicas is None else num_replicas
            self.rank = torch.distributed.get_rank() if rank is None else rank

        self.drop_last = drop_last
        
        self.total_size = len(self.dataset)

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on seed
            rng = np.random.default_rng(seed=self.seed)
            indices = rng.choice(len(self.dataset), size=len(self.dataset),
                replace=False).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples
