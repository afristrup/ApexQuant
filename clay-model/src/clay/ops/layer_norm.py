"""Layer normalisation, fused on CUDA and torch elsewhere."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .use_triton import use_triton


def layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    """Normalise the last dimension of ``x`` and apply the affine transform.

    Args:
        x: Input of shape ``(..., N)``.
        weight: Affine gain of shape ``(N,)``.
        bias: Affine shift of shape ``(N,)``.
        eps: Variance floor.

    Returns:
        Tensor shaped like ``x``.
    """
    if use_triton(x):
        from ..kernels.triton_layer_norm import triton_layer_norm

        return triton_layer_norm(x, weight, bias, eps)
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
