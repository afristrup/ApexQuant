"""Reconstruct a dense float weight from a QuantizedWeight (reference path)."""

from __future__ import annotations

import torch

from ..rotation.inverse_guarded_rotation import inverse_guarded_rotation
from ..rotation.inverse_random_rotation import inverse_random_rotation
from .quantized_weight import QuantizedWeight
from .unpack_bits import unpack_bits


def dequantize_weight(qw: QuantizedWeight) -> torch.Tensor:
    """Rebuild the approximate dense weight ``(out_features, in_features)``.

    This is the mathematical inverse of :func:`turboquantize_weight`: for every
    pass, look up the codebook, undo that pass's rotation, and accumulate. It is
    used for evaluation, tests, and any code path that wants a plain tensor. The
    fused kernels avoid ever calling this.

    Args:
        qw: The quantized weight.

    Returns:
        Dense float32 weight with the input padding sliced back off.
    """
    out = torch.zeros(
        qw.out_features, qw.num_groups, qw.group_size,
        dtype=torch.float32, device=qw.device,
    )
    for p in range(qw.passes):
        seed = qw.seed + p
        indices = unpack_bits(qw.packed[p], qw.bits).long()  # (out, groups, gs)
        recon_rot = qw.scale[p].unsqueeze(-1) * qw.levels[indices]
        if qw.guard is None:
            out = out + inverse_random_rotation(recon_rot, seed)
        else:
            out = out + inverse_guarded_rotation(recon_rot, seed, qw.guard[p])

    dense = out.reshape(qw.out_features, qw.num_groups * qw.group_size)
    dense = dense[:, : qw.in_features]

    # Undo the Sinkhorn preprocessing, if any. It was applied before the first
    # rotation, so it comes off last; the packed passes above reconstruct the
    # *normalized* matrix.
    if qw.row_scale is not None:
        dense = dense * qw.row_scale.to(dense.dtype).unsqueeze(1)
    if qw.col_scale is not None:
        dense = dense * qw.col_scale.to(dense.dtype)
    return dense.contiguous()
