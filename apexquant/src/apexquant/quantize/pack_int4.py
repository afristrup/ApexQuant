"""Pack 4-bit indices two-per-byte."""

from __future__ import annotations

import torch


def pack_int4(indices: torch.Tensor) -> torch.Tensor:
    """Pack a uint8 tensor of 4-bit values into half as many bytes.

    Even positions go in the low nibble, odd positions in the high nibble, so a
    length-``d`` last axis becomes length ``d // 2``. This is the packing that
    turns "16 possible values" into the advertised 4x memory saving over bf16.

    Args:
        indices: uint8 tensor with values in ``[0, 16)`` and an even last dim.

    Returns:
        uint8 tensor with the last dim halved.
    """
    if indices.shape[-1] % 2 != 0:
        raise ValueError("last dimension must be even to nibble-pack")
    low = indices[..., 0::2]
    high = indices[..., 1::2]
    return (low | (high << 4)).to(torch.uint8)
