"""Invert the guarded randomized Hadamard rotation."""

from __future__ import annotations

import torch

from .guarded_rotation import GUARD_SEED_OFFSET
from .inverse_random_rotation import inverse_random_rotation


def inverse_guarded_rotation(y: torch.Tensor, seed: int, flags: torch.Tensor) -> torch.Tensor:
    """Undo :func:`guarded_rotation` given the flags it returned."""
    if flags.any():
        once = inverse_random_rotation(y, seed + GUARD_SEED_OFFSET)
        y = torch.where(flags.unsqueeze(-1), once, y)
    return inverse_random_rotation(y, seed)
