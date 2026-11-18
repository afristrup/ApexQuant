"""Launch the column-scaled fused Walsh-Hadamard rotation over all passes."""

from __future__ import annotations

import math

import torch
import triton

from .triton_wht_rotate import MAX_FUSED_DIM
from .wht_rotate_scaled_kernel import wht_rotate_scaled_kernel
from .wht_row_tile import wht_row_tile


def triton_wht_rotate_scaled(x: torch.Tensor, col: torch.Tensor, dim: int,
                             signs: torch.Tensor) -> torch.Tensor:
    """``triton_wht_rotate`` of ``x * col`` without materializing the product.

    Args:
        x: Activations ``(tokens, groups * dim)``, contiguous.
        col: Per-input-column scale ``(groups * dim,)``, float32.
        dim: Group size, a power of two.
        signs: ``(passes, dim)`` sign vectors.

    Returns:
        Tensor of shape ``(passes, tokens * groups, dim)``.
    """
    tokens, width = x.shape
    if width % dim or col.numel() != width:
        raise ValueError(f"width {width}, column scale {col.numel()}, group {dim} disagree")
    if dim & (dim - 1) or dim > MAX_FUSED_DIM:
        raise ValueError(f"group size {dim} must be a power of two <= {MAX_FUSED_DIM}")
    groups, passes = width // dim, signs.shape[0]
    rows = tokens * groups
    out = torch.empty(passes, rows, dim, device=x.device, dtype=x.dtype)
    tile = wht_row_tile(dim)
    wht_rotate_scaled_kernel[(triton.cdiv(rows, tile), passes)](
        x, col, signs, out, rows, groups, rows * dim, 1.0 / math.sqrt(dim),
        D=dim, LOG_D=int(math.log2(dim)), BLOCK_R=tile,
    )
    return out
