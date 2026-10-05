"""Sign vectors for a run of consecutive rotation seeds, built once."""

from __future__ import annotations

import functools

import torch

from .random_sign_vector import random_sign_vector


@functools.lru_cache(maxsize=None)
def stacked_signs(dim: int, seed: int, passes: int, device: str) -> torch.Tensor:
    """Return ``(passes, dim)`` sign vectors for seeds ``seed..seed + passes - 1``.

    Cached because a fused rotation reads it once per layer per forward, and
    restacking it there is an allocation and a copy on the critical path.

    Args:
        dim: Group size.
        seed: Base seed; pass ``p`` uses ``seed + p``.
        passes: Number of residual passes.
        device: Device string, part of the cache key.

    Returns:
        Contiguous float32 tensor of shape ``(passes, dim)``.
    """
    return torch.stack([
        random_sign_vector(dim, seed + p, device=torch.device(device)).float()
        for p in range(passes)
    ]).contiguous()
