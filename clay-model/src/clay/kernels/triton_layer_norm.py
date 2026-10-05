"""Launch the fused layer-normalisation kernel."""

from __future__ import annotations

import torch
import triton

from .layer_norm_kernel import layer_norm_kernel

# Rows wider than this would need a multi-pass kernel; Clay's widest normalised
# tensor is the 1024-dim large encoder, so a single block always suffices.
MAX_FUSED_WIDTH = 8192


def triton_layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    """Layer-normalise the last dimension of ``x`` with a single Triton pass.

    Args:
        x: Input of shape ``(..., N)``.
        weight: Affine gain of shape ``(N,)``.
        bias: Affine shift of shape ``(N,)``.
        eps: Added to the variance before the reciprocal square root.

    Returns:
        Tensor shaped like ``x``.
    """
    n_cols = x.shape[-1]
    if n_cols > MAX_FUSED_WIDTH:
        raise ValueError(
            f"row width {n_cols} exceeds the fused limit {MAX_FUSED_WIDTH}"
        )

    x_flat = x.reshape(-1, n_cols).contiguous()
    y_flat = torch.empty_like(x_flat)
    layer_norm_kernel[(x_flat.shape[0],)](
        x_flat,
        y_flat,
        weight,
        bias,
        x_flat.stride(0),
        n_cols,
        eps,
        BLOCK_N=triton.next_power_of_2(n_cols),
    )
    return y_flat.view_as(x)
