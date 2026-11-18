"""Randomized Hadamard rotation with a second transform for spiky groups."""

from __future__ import annotations

import torch

from .normalized_third_moment import normalized_third_moment
from .random_rotation import random_rotation

GUARD_SEED_OFFSET = 1_000_003


def guarded_rotation(x: torch.Tensor, seed: int, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate ``x`` and apply a second independent rotation where it is spiky.

    A group is flagged when its normalized third moment exceeds ``threshold``.
    The flag depends only on the group being rotated, so nested prefixes stay
    identical.

    Returns:
        The rotated tensor and a bool flag per group (shape ``x.shape[:-1]``).
    """
    flags = normalized_third_moment(x) > threshold
    rotated = random_rotation(x, seed)
    if flags.any():
        twice = random_rotation(rotated, seed + GUARD_SEED_OFFSET)
        rotated = torch.where(flags.unsqueeze(-1), twice, rotated)
    return rotated, flags
