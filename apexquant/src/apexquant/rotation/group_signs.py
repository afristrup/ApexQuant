"""Independent sign vectors for every group of a pass, built once."""

from __future__ import annotations

import functools

import torch

from .random_sign_vector import random_sign_vector


@functools.lru_cache(maxsize=64)
def group_signs(groups: int, dim: int, seed: int, device: str) -> torch.Tensor:
    """``(groups, dim)`` signs; group ``g`` of pass seed ``seed`` uses ``seed * 1_000_003 + g``."""
    return torch.stack([random_sign_vector(dim, seed * 1_000_003 + g) for g in range(groups)]
                       ).to(device)
