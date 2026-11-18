"""Residual round-to-nearest passes in the original coordinates, reading no data."""

from __future__ import annotations

import torch


@torch.no_grad()
def rtn_residual(weight: torch.Tensor, bits: int, passes: int, group: int) -> torch.Tensor:
    """``passes`` asymmetric min-max RTN passes on the running residual, per group.

    This is ApexQuant's recursion with the fresh rotation and the spherical stage
    removed, so at matched bytes it isolates what the rotation contributes.
    """
    out_f, in_f = weight.shape
    if in_f % group:
        raise ValueError(f"fan-in {in_f} not divisible by group {group}")
    w = weight.float().view(out_f, -1, group)
    acc = torch.zeros_like(w)
    top = 2 ** bits - 1
    for _ in range(passes):
        r = w - acc
        lo, hi = r.amin(-1, keepdim=True), r.amax(-1, keepdim=True)
        scale = ((hi - lo) / top).clamp_min(1e-12)
        acc = acc + ((r - lo) / scale).round().clamp(0, top) * scale + lo
    return acc.view(out_f, in_f).to(weight.dtype)
