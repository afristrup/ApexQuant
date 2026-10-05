"""Unpack two-per-byte 4-bit indices back to uint8."""

from __future__ import annotations

import torch


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_int4`; doubles the last dimension.

    Args:
        packed: uint8 tensor produced by :func:`pack_int4`.

    Returns:
        uint8 tensor of indices in ``[0, 16)`` with the last dim doubled, values
        re-interleaved as ``[low0, high0, low1, high1, ...]``.
    """
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = torch.stack([low, high], dim=-1)  # (..., d/2, 2)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2).to(torch.uint8)
