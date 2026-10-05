"""Unpack bytes back into low-bit indices (inverse of pack_bits)."""

from __future__ import annotations

import torch


def unpack_bits(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Inverse of :func:`pack_bits`; scales the last dimension by ``8 // bits``.

    Args:
        packed: uint8 tensor produced by :func:`pack_bits`.
        bits: 2, 4, or 8 (must match the packing).

    Returns:
        uint8 tensor of indices in ``[0, 2**bits)`` with the last dim scaled by
        ``8 // bits``, positions re-interleaved low-bits-first per byte.
    """
    if 8 % bits != 0:
        raise ValueError("bits must divide 8 (2, 4, or 8)")
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    shifts = (bits * torch.arange(per_byte, device=packed.device, dtype=torch.int32))
    vals = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & mask  # (..., L, per_byte)
    return vals.reshape(*packed.shape[:-1], packed.shape[-1] * per_byte).to(torch.uint8)
