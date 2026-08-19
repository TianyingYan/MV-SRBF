# encoding: utf-8
import math
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

class RandomIDOKSampler(Sampler):
    """
    Random O/K identity sampler for large datasets with optional aux balancing.

    Features:
      - Global OID sampling balanced by inverse frequency.
      - Per-OID aux combination sampling balanced by inverse frequency (if aux_dims given)
      - DDP-safe: each rank independently samples
      - Epoch length auto-computed from data size, O, K, world_size.
    """

    def __init__(
            self,
            dataset,
            o_ids_per_gpu,
            k_instances_per_id,
            aux_dims=None,
            num_replicas=None,
            rank=None,
            drop_last=True,
            shuffle=True,
            seed=0,
    ):
        if num_replicas is None:
            self.num_replicas = (
                torch.distributed.get_world_size()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 1
            )
        else:
            self.num_replicas = int(num_replicas)
        if rank is None:
            self.rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 0
            )
        else:
            self.rank = int(rank)

        self.dataset = dataset
        self.o = o_ids_per_gpu
        self.k = k_instances_per_id
        self.aux_dims = aux_dims if aux_dims is not None else []
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.oid_to_indices = defaultdict(list)
        if self.aux_dims:
            self.oid_to_aux_groups = defaultdict(lambda: defaultdict(list))

        metadata = getattr(self.dataset, "data_list", self.dataset)
        for idx, item in enumerate(metadata):
            if len(item) < 2:
                raise ValueError(f"Item {idx} must have at least (img_meta, oid). Got: {item}")
            oid = item[1]
            self.oid_to_indices[oid].append(idx)

            if self.aux_dims:
                aux_vals = []
                for i in self.aux_dims:
                    pos = 2 + i
                    if pos < len(item):
                        aux_vals.append(item[pos])
                    else:
                        raise IndexError(
                            f"Item {idx} has length {len(item)}, but aux_dim {i} requires position {pos}."
                        )
                aux_tuple = tuple(aux_vals)
                self.oid_to_aux_groups[oid][aux_tuple].append(idx)

        self.all_oids = list(self.oid_to_indices.keys())
        if len(self.all_oids) < self.o:
            raise ValueError(f"Number of identities ({len(self.all_oids)}) < o_ids_per_gpu ({self.o})")

        counts_oid = np.array([len(self.oid_to_indices[oid]) for oid in self.all_oids], dtype=np.float64)
        weights_oid = 1.0 / (counts_oid + 1e-8)
        self.oid_prob = weights_oid / weights_oid.sum()

        if self.aux_dims:
            self.oid_to_aux_prob = {}
            for oid in self.all_oids:
                aux_combos = list(self.oid_to_aux_groups[oid].keys())
                if not aux_combos:
                    continue
                counts_aux = np.array([
                    len(self.oid_to_aux_groups[oid][combo]) for combo in aux_combos
                ], dtype=np.float64)
                weights_aux = 1.0 / (counts_aux + 1e-8)
                prob_aux = weights_aux / weights_aux.sum()
                self.oid_to_aux_prob[oid] = (aux_combos, prob_aux)

        # One epoch is dataset_size / (world_size * O * K) optimizer steps.
        total_samples = len(self.dataset)
        per_step = self.o * self.k
        denom = self.num_replicas * per_step
        if self.drop_last:
            self.num_batches = max(1, total_samples // denom)
        else:
            self.num_batches = max(1, math.ceil(total_samples / denom))

    def __iter__(self):
        if self.shuffle:
            rng = np.random.default_rng(seed=self.seed + self.epoch + self.rank)
        else:
            rng = np.random.default_rng(seed=self.seed + self.rank)  # fixed across epochs

        final_indices = []

        for _ in range(self.num_batches):
            # Sample O distinct OIDs with balanced probability
            selected_oids = rng.choice(
                self.all_oids,
                size=self.o,
                replace=False,
                p=self.oid_prob
            )

            batch_indices = []
            for oid in selected_oids:
                if self.aux_dims:
                    aux_combos, aux_prob = self.oid_to_aux_prob[oid]
                    n_combos = len(aux_combos)

                    if n_combos >= self.k:
                        chosen_combos = rng.choice(
                            aux_combos,
                            size=self.k,
                            replace=False,
                            p=aux_prob
                        )
                    else:
                        chosen_combos = rng.choice(
                            aux_combos,
                            size=self.k,
                            replace=True,
                            p=aux_prob
                        )

                    for combo in chosen_combos:
                        if isinstance(combo, np.ndarray):
                            combo = tuple(combo.tolist())
                        elif not isinstance(combo, tuple):
                            combo = (combo,)
                        pool = self.oid_to_aux_groups[oid][combo]
                        idx = rng.choice(pool, size=1, replace=True).item()
                        batch_indices.append(idx)

                else:
                    pool = self.oid_to_indices[oid]
                    n_samples = len(pool)
                    if n_samples >= self.k:
                        sampled = rng.choice(pool, size=self.k, replace=False)
                    else:
                        sampled = rng.choice(pool, size=self.k, replace=True)
                    batch_indices.extend(sampled.tolist())

            final_indices.extend(batch_indices)

        return iter(final_indices)

    def __len__(self):
        return self.num_batches * self.o * self.k

    def set_epoch(self, epoch):
        self.epoch = epoch


class UniqueIDOKSampler(Sampler):
    """
    Dataset-covering O/K identity sampler for mini datasets.

    Unlike ``RandomIDOKSampler``, this sampler first builds shuffled K-instance
    chunks for each identity and then consumes those chunks across the epoch.
    This better matches conventional RandomIdentitySampler behavior: small
    datasets are traversed more completely instead of repeatedly drawing a
    dataset-sized number of random O/K batches.
    """

    def __init__(
        self,
        dataset,
        o_ids_per_gpu,
        k_instances_per_id,
        aux_dims=None,
        num_replicas=None,
        rank=None,
        drop_last=True,
        shuffle=True,
        seed=0,
    ):
        if num_replicas is None:
            self.num_replicas = (
                torch.distributed.get_world_size()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 1
            )
        else:
            self.num_replicas = int(num_replicas)
        if rank is None:
            self.rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 0
            )
        else:
            self.rank = int(rank)

        self.dataset = dataset
        self.o = int(o_ids_per_gpu)
        self.k = int(k_instances_per_id)
        self.aux_dims = aux_dims if aux_dims is not None else []
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        self.oid_to_indices = defaultdict(list)
        self.oid_to_aux_groups = defaultdict(lambda: defaultdict(list))
        metadata = getattr(self.dataset, "data_list", self.dataset)
        for idx, item in enumerate(metadata):
            if len(item) < 2:
                raise ValueError(f"Item {idx} must have at least (img_meta, oid). Got: {item}")
            oid = item[1]
            self.oid_to_indices[oid].append(idx)
            if self.aux_dims:
                aux_vals = []
                for dim in self.aux_dims:
                    pos = 2 + dim
                    if pos >= len(item):
                        raise IndexError(
                            f"Item {idx} has length {len(item)}, but aux_dim {dim} requires position {pos}."
                        )
                    aux_vals.append(item[pos])
                self.oid_to_aux_groups[oid][tuple(aux_vals)].append(idx)

        self.all_oids = list(self.oid_to_indices.keys())
        if len(self.all_oids) < self.o:
            raise ValueError(f"Number of identities ({len(self.all_oids)}) < o_ids_per_gpu ({self.o})")
        if len(self.all_oids) < self.o * self.num_replicas:
            raise ValueError(
                "UniqueIDOKSampler requires at least "
                f"world_size * o_ids_per_gpu identities ({self.o * self.num_replicas}), "
                f"got {len(self.all_oids)}."
            )

    def _rng(self) -> np.random.Generator:
        # All DDP ranks build the same global plan, then each rank takes its block.
        seed = self.seed + (self.epoch if self.shuffle else 0)
        return np.random.default_rng(seed)

    def _pid_chunks(self, rng: np.random.Generator) -> Dict[int, List[List[int]]]:
        chunks_by_oid: Dict[int, List[List[int]]] = {}
        for oid in self.all_oids:
            if self.aux_dims:
                chunks = self._aux_balanced_chunks(oid, rng)
            else:
                chunks = self._plain_chunks(self.oid_to_indices[oid], rng)
            if chunks:
                chunks_by_oid[oid] = chunks
        return chunks_by_oid

    def _plain_chunks(self, indices: List[int], rng: np.random.Generator) -> List[List[int]]:
        idxs = list(indices)
        if len(idxs) < self.k:
            return [rng.choice(idxs, size=self.k, replace=True).astype(int).tolist()]
        if self.shuffle:
            rng.shuffle(idxs)
        usable = len(idxs) - (len(idxs) % self.k)
        if usable <= 0:
            return []
        return [idxs[i:i + self.k] for i in range(0, usable, self.k)]

    def _aux_balanced_chunks(self, oid: int, rng: np.random.Generator) -> List[List[int]]:
        groups = {combo: list(indices) for combo, indices in self.oid_to_aux_groups[oid].items()}
        for indices in groups.values():
            if self.shuffle:
                rng.shuffle(indices)
        total = sum(len(indices) for indices in groups.values())
        if total < self.k:
            pool = [idx for indices in groups.values() for idx in indices]
            return [rng.choice(pool, size=self.k, replace=True).astype(int).tolist()]

        chunks: List[List[int]] = []
        while sum(len(indices) for indices in groups.values()) >= self.k:
            active = [combo for combo, indices in groups.items() if indices]
            if self.shuffle:
                rng.shuffle(active)
            chunk: List[int] = []
            for combo in active:
                if len(chunk) >= self.k:
                    break
                chunk.append(groups[combo].pop())
            while len(chunk) < self.k:
                active = [combo for combo, indices in groups.items() if indices]
                if not active:
                    break
                combo = active[int(rng.integers(0, len(active)))]
                chunk.append(groups[combo].pop())
            if len(chunk) == self.k:
                chunks.append(chunk)
        return chunks

    def _fresh_plain_chunk(self, oid: int, rng: np.random.Generator) -> List[int]:
        pool = self.oid_to_indices[oid]
        replace = len(pool) < self.k
        return rng.choice(pool, size=self.k, replace=replace).astype(int).tolist()

    def _fresh_aux_chunk(self, oid: int, rng: np.random.Generator) -> List[int]:
        groups = self.oid_to_aux_groups[oid]
        combos = list(groups.keys())
        if self.shuffle:
            rng.shuffle(combos)
        chunk: List[int] = []
        for combo in combos:
            if len(chunk) >= self.k:
                break
            chunk.append(int(rng.choice(groups[combo], size=1, replace=True).item()))
        while len(chunk) < self.k:
            combo = combos[int(rng.integers(0, len(combos)))]
            chunk.append(int(rng.choice(groups[combo], size=1, replace=True).item()))
        return chunk

    def _fresh_chunk(self, oid: int, rng: np.random.Generator) -> List[int]:
        if self.aux_dims:
            return self._fresh_aux_chunk(oid, rng)
        return self._fresh_plain_chunk(oid, rng)

    def _sample_indices_for_epoch(self) -> List[int]:
        rng = self._rng()
        chunks_by_oid = self._pid_chunks(rng)
        available = [oid for oid in self.all_oids if oid in chunks_by_oid]
        global_o = self.o * self.num_replicas
        final_indices: List[int] = []

        while len(available) >= global_o:
            if self.shuffle:
                selected = rng.choice(available, size=global_o, replace=False).tolist()
            else:
                selected = available[:global_o]

            global_batch: List[int] = []
            exhausted = []
            for oid in selected:
                global_batch.extend(chunks_by_oid[oid].pop(0))
                if not chunks_by_oid[oid]:
                    exhausted.append(oid)
            for oid in exhausted:
                if oid in available:
                    available.remove(oid)

            start = self.rank * self.o * self.k
            end = start + self.o * self.k
            final_indices.extend(global_batch[start:end])

        if available and not self.drop_last:
            selected = list(available)
            pad_needed = global_o - len(selected)
            pad_pool = [oid for oid in self.all_oids if oid not in selected]
            replace = len(pad_pool) < pad_needed
            selected.extend([int(oid) for oid in rng.choice(pad_pool, size=pad_needed, replace=replace).tolist()])

            global_batch = []
            for oid in selected:
                if oid in available and chunks_by_oid.get(oid):
                    global_batch.extend(chunks_by_oid[oid].pop(0))
                else:
                    global_batch.extend(self._fresh_chunk(oid, rng))

            start = self.rank * self.o * self.k
            end = start + self.o * self.k
            final_indices.extend(global_batch[start:end])

        return final_indices

    def __iter__(self):
        return iter(self._sample_indices_for_epoch())

    def __len__(self):
        return len(self._sample_indices_for_epoch())

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
