"""Linear projection with a GELU epilogue, fused on CUDA and torch elsewhere."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .use_triton import use_triton


def linear_gelu(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Return ``gelu(x @ weight.T + bias)`` using the exact (erf) GELU.

    Args:
        x: Input of shape ``(..., n_in)``.
        weight: ``nn.Linear`` weight of shape ``(n_out, n_in)``.
        bias: ``nn.Linear`` bias of shape ``(n_out,)``.

    Returns:
        Tensor of shape ``(..., n_out)``.
    """
    if use_triton(x):
        from ..kernels.triton_linear_gelu import triton_linear_gelu

        return triton_linear_gelu(x, weight, bias)
    return F.gelu(F.linear(x, weight, bias))
