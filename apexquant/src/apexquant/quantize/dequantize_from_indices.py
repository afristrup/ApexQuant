"""Codebook lookup: indices -> reconstruction levels."""

from __future__ import annotations

import torch


def dequantize_from_indices(indices: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Gather codebook ``levels`` at ``indices``.

    Args:
        indices: Integer index tensor (any shape, any integer dtype).
        levels: 1-D codebook of reconstruction values.

    Returns:
        Float tensor of the same shape as ``indices``, dtype of ``levels``.
    """
    return levels[indices.long()]
