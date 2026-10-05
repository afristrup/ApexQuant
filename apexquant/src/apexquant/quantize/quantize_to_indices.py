"""Nearest-codebook-level assignment."""

from __future__ import annotations

import torch


def quantize_to_indices(x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Map each element of ``x`` to the index of its nearest codebook level.

    ``levels`` is sorted, so the nearest level is found in ``O(log n)`` per
    element with a binary search over the decision boundaries (the midpoints
    between adjacent levels) instead of an ``O(n)`` distance scan.

    Args:
        x: Values to quantize (any shape).
        levels: Sorted 1-D codebook, length ``2**bits``.

    Returns:
        uint8 tensor of indices, same shape as ``x``.
    """
    boundaries = 0.5 * (levels[1:] + levels[:-1])
    idx = torch.bucketize(x.contiguous(), boundaries.to(x.dtype))
    return idx.to(torch.uint8)
