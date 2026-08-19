from .training_sampler import RandomIDOKSampler, UniqueIDOKSampler
from .inference_sampler import InferenceSampler

def build_training_sampler(cfg, dataset):
    train_sampler_cfg = cfg["dataloader"]["train"]["sampler"]

    name = train_sampler_cfg["name"].lower()
    o = train_sampler_cfg["o_ids_per_gpu"]
    k = train_sampler_cfg["k_instances_per_id"]
    aux_dims = train_sampler_cfg.get("aux_dims")

    if name not in {"random", "unique"}:
        raise ValueError(f"Unsupported sampler name: {name}. Use random or unique.")

    if o < 2:
        raise ValueError(f"sampler '{name}' requires o_ids_per_gpu >= 2")
    if k < 2:
        raise ValueError(f"sampler '{name}' requires k_instances_per_id >= 2")

    common_kwargs = {
        "dataset": dataset,
        "o_ids_per_gpu": o,
        "k_instances_per_id": k,
        "aux_dims": aux_dims,
        "drop_last": bool(cfg["dataloader"]["train"].get("drop_last", True)),
        "shuffle": train_sampler_cfg["shuffle"],
        "seed": cfg["seed"],
    }
    if name == "unique":
        return UniqueIDOKSampler(**common_kwargs)
    return RandomIDOKSampler(**common_kwargs)


def build_query_sampler(cfg, dataset):
    query_cfg = cfg["dataloader"]["query"]["sampler"]
    # Any-modal evaluation runs only on rank 0 and does not gather features across
    # ranks, so the query set must be covered in full on rank 0. Pin the sampler to
    # a single replica instead of letting InferenceSampler shard it per DDP rank.
    return InferenceSampler(
        dataset=dataset,
        num_replicas=1,
        rank=0,
        shuffle=query_cfg["shuffle"],
        seed=cfg["seed"],
        drop_last=query_cfg["drop_last"]
    )
