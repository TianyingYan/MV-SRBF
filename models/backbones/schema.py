from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class BackboneOutput:
    cls_pre: torch.Tensor
    spatial_tokens: Optional[torch.Tensor] = None
    spatial_hw: Optional[Tuple[int, int]] = None
