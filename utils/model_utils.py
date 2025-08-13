"""Model utilities: parameter counting and dummy input creation."""
from __future__ import annotations

from typing import Tuple

import torch


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def dummy_input(nchw: Tuple[int, int, int, int] = (1, 3, 32, 32), device: str | torch.device = "cpu") -> torch.Tensor:
    return torch.randn(*nchw, device=device)
