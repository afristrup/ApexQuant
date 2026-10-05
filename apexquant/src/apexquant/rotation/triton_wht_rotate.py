"""Launch the fused Walsh-Hadamard rotation over all passes."""

from __future__ import annotations

import math

import torch
import triton

from .stacked_signs import stacked_signs
from .wht_rotate_kernel import wht_rotate_kernel
from .wht_row_tile import wht_row_tile

# The transform is held in registers a tile at a time, so the group has to fit.
MAX_FUSED_DIM = 1024


def triton_wht_rotate(x: torch.Tensor, dim: int, seed: int,
                      passes: int) -> torch.Tensor:
    """Rotate grouped rows by each pass's randomized Hadamard, in one launch.

    Same result as ``torch.matmul`` against the cached rotation matrices, to
    floating-point rounding, at ``dim log dim`` work instead of ``dim**2``. On a
    4090 over Clay's rotation shapes that is 1.4x at g=128 and 4.8x at g=512,
    and the cost is nearly flat in the group size -- the transform moves the same
    bytes whatever the group, and only the arithmetic grew.

    Args:
        x: Grouped rows ``(rows, dim)``, contiguous.
        dim: Group size, a power of two.
        seed: Base sign seed; pass ``p`` uses ``seed + p``.
        passes: Number of residual passes.

    Returns:
        Tensor of shape ``(passes, rows, dim)``.
    """
    rows, width = x.shape
    if width != dim:
        raise ValueError(f"row width {width} does not match group size {dim}")
    if dim & (dim - 1) or dim > MAX_FUSED_DIM:
        raise ValueError(f"group size {dim} must be a power of two <= {MAX_FUSED_DIM}")

    signs = stacked_signs(dim, seed, passes, str(x.device))
    out = torch.empty(passes, rows, dim, device=x.device, dtype=x.dtype)
    tile = wht_row_tile(dim)
    wht_rotate_kernel[(triton.cdiv(rows, tile), passes)](
        x, signs, out, rows, rows * dim, 1.0 / math.sqrt(dim),
        D=dim, LOG_D=int(math.log2(dim)), BLOCK_R=tile,
    )
    return out
