"""Reference fused dequant+matmul in pure torch (CPU/GPU, always available)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..quantize.quantized_weight import QuantizedWeight
from ..quantize.unpack_bits import unpack_bits
from ..rotation.random_rotation import random_rotation


def torch_fused_dequant_matmul(x: torch.Tensor, qw: QuantizedWeight) -> torch.Tensor:
    """Compute ``x @ dequantize_weight(qw).T`` without materializing the weight.

    The ApexQuant identity ``<x, R^T c> == <R x, c>`` lets us rotate the *input*
    once per pass and contract it against the codebook lookup ``levels[idx]``.
    The float weight is never formed: the only "dequant" is a gather of 16 scalar
    levels, fused into the group contraction. This is the semantics the Triton
    kernel implements at speed; here it is expressed as a batched einsum so it is
    obviously correct and runs anywhere torch runs.

    Args:
        x: Input activations ``(..., in_features)``.
        qw: The quantized weight.

    Returns:
        Output ``(..., out_features)``.
    """
    if qw.device != x.device:  # allow a CPU-quantized weight against GPU input
        qw = qw.to(x.device)
    lead = x.shape[:-1]
    gs = qw.group_size
    x2 = x.reshape(-1, x.shape[-1])                       # (M, in)
    pad = qw.num_groups * gs - x2.shape[-1]
    xg = F.pad(x2, (0, pad)).view(-1, qw.num_groups, gs)  # (M, groups, gs)

    levels = qw.levels.to(x.dtype)
    out = x.new_zeros(x2.shape[0], qw.out_features)
    rotations = qw.rotations
    for p in range(qw.passes):
        if rotations is None:
            rx = random_rotation(xg, qw.seed + p)
        else:
            rx = torch.einsum("mng,gh->mnh", xg, rotations[p].to(x.dtype))
        codes = levels[unpack_bits(qw.packed[p], qw.bits).long()]   # (out, groups, gs)
        # Fold the per-(out, group) scale into the codebook values so the whole
        # pass is a single contraction -- no (M, out, groups) intermediate.
        codes = codes * qw.scale[p].to(x.dtype).unsqueeze(-1)       # (out, groups, gs)
        out = out + torch.einsum("mgd,ogd->mo", rx, codes)         # (M, out)

    return out.view(*lead, qw.out_features)
