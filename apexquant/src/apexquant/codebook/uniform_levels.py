"""Uniform (fixed-grid) codebook -- the QuaRot / RTN post-rotation baseline."""

from __future__ import annotations

import torch


def uniform_levels(bits: int, device=None) -> torch.Tensor:
    """Return a symmetric mid-rise uniform codebook of ``2**bits`` levels on ``[-1, 1]``.

    This is the grid that QuaRot and round-to-nearest place after rotation --
    the baseline ApexQuant improves on. Having it as a first-class codebook lets
    the unified framework reproduce those methods exactly for apples-to-apples
    comparison. Pairs with absmax normalization (values land in ``[-1, 1]``).

    Args:
        bits: Codebook bit-width; ``2**bits`` levels.
        device: Optional device for the returned tensor.

    Returns:
        Sorted 1-D float32 tensor of ``2**bits`` levels, symmetric about 0.
    """
    n = 2 ** bits
    # Mid-rise: cell centers of a uniform partition of [-1, 1].
    idx = torch.arange(n, dtype=torch.float32, device=device)
    return (2.0 * idx + 1.0) / n - 1.0
