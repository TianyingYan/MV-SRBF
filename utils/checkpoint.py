from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer,
    epoch: int,
    metadata: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a trainer checkpoint in the project-standard format.

    Args:
        path: Destination ``.pth`` file.
        model: Model whose ``state_dict`` is stored under ``"model"``.
        optimizer: Optimizer whose ``state_dict`` is stored under ``"optimizer"``.
        epoch: Bookkeeping step (the trainer passes the global optimizer-update count).
        metadata: Optional model/backbone descriptor stored under ``"metadata"``.
        extra: Optional extra keys merged into the payload. The trainer uses this to
            store ``"trainer_state"`` (scheduler, AMP scaler, counters, best metrics,
            and early-stop state) so that breakpoint training can resume exactly.
    """
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": metadata or {},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
