"""Pack low-bit indices into bytes (generalizes nibble packing to any width)."""

from __future__ import annotations

import torch


def pack_bits(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack ``8 // bits`` indices per byte along the last dimension.

    ``bits=4`` reproduces :func:`pack_int4` exactly (two per byte); ``bits=2``
    packs four per byte (the 16x-compression regime where the density-matched
    codebook matters most); ``bits=8`` is one per byte. Value at position ``j``
    within a byte occupies bit range ``[j*bits, (j+1)*bits)``, so position 0 is the
    low bits -- the same convention the kernels decode.

    Args:
        indices: uint8 tensor with values in ``[0, 2**bits)`` and a last dimension
            divisible by ``8 // bits``.
        bits: 2, 4, or 8.

    Returns:
        uint8 tensor with the last dim scaled by ``bits / 8``.
    """
    if 8 % bits != 0:
        raise ValueError("bits must divide 8 (2, 4, or 8)")
    per_byte = 8 // bits
    d = indices.shape[-1]
    if d % per_byte != 0:
        raise ValueError(f"last dim {d} must be divisible by {per_byte} to pack {bits}-bit")

    grouped = indices.to(torch.int32).reshape(*indices.shape[:-1], d // per_byte, per_byte)
    shifts = (bits * torch.arange(per_byte, device=indices.device, dtype=torch.int32))
    return (grouped << shifts).sum(dim=-1).to(torch.uint8)
