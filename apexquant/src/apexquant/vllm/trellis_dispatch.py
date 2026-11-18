"""Graph-safe runtime dispatch for packed Trellis linears."""

from __future__ import annotations

import os

import torch

from ..kernels.triton_gemm_trellis import triton_gemm_trellis
from ..kernels.triton_gemv_trellis import triton_gemv_trellis


@torch.library.custom_op(
    "apexquant::trellis_rotate",
    mutates_args=(),
    device_types="cuda",
)
def trellis_rotate(
    x: torch.Tensor,
    signs: torch.Tensor,
    group_size: int,
    seed: int,
    passes: int,
) -> torch.Tensor:
    """Use FP16 WHT for one-token decode when explicitly enabled."""
    from ..rotation.triton_wht_rotate import triton_wht_rotate

    source = x if (x.shape[0] == 1 and os.environ.get(
        "APEXQUANT_TRELLIS_FP16_ROTATE") == "1") else x.float()
    rotated = triton_wht_rotate(source.reshape(-1, group_size), group_size,
                                seed, passes, signs=signs)
    rotated = rotated.view(passes, x.shape[0], -1)
    return rotated if source.dtype == x.dtype else rotated.float()


@trellis_rotate.register_fake
def _trellis_rotate_fake(
    x: torch.Tensor,
    signs: torch.Tensor,
    group_size: int,
    seed: int,
    passes: int,
) -> torch.Tensor:
    dtype = x.dtype if os.environ.get("APEXQUANT_TRELLIS_FP16_ROTATE") == "1" else torch.float32
    return x.new_empty((passes, x.shape[0], x.shape[1]), dtype=dtype)


@torch.library.custom_op(
    "apexquant::trellis_linear",
    mutates_args=(),
    device_types="cuda",
)
def trellis_linear(
    rotated: torch.Tensor,
    words: torch.Tensor,
    scale: torch.Tensor,
    table: torch.Tensor,
    group_size: int,
    bits: int,
    trellis_bits: int,
) -> torch.Tensor:
    """Select GEMV at decode and GEMM for multi-token prefill."""
    args = (words, scale, table, group_size, bits, trellis_bits)
    if rotated.shape[1] == 1:
        return triton_gemv_trellis(rotated[:, 0], *args)
    return triton_gemm_trellis(rotated, *args)


@trellis_linear.register_fake
def _trellis_linear_fake(
    rotated: torch.Tensor,
    words: torch.Tensor,
    scale: torch.Tensor,
    table: torch.Tensor,
    group_size: int,
    bits: int,
    trellis_bits: int,
) -> torch.Tensor:
    return rotated.new_empty((rotated.shape[1], words.shape[1]), dtype=torch.float32)
