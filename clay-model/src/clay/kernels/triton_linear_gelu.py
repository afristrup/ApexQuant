"""Launch the fused linear + GELU kernel."""

from __future__ import annotations

import torch
import triton

from ..constants import TRITON_DOT_PRECISION
from .linear_gelu_kernel import linear_gelu_kernel


def triton_linear_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    precision: str = TRITON_DOT_PRECISION,
) -> torch.Tensor:
    """Return ``gelu(x @ weight.T + bias)`` without materialising the pre-activation.

    Args:
        x: Input of shape ``(..., n_in)``.
        weight: ``nn.Linear`` weight of shape ``(n_out, n_in)``.
        bias: ``nn.Linear`` bias of shape ``(n_out,)``.
        precision: ``tl.dot`` accumulation mode, "tf32" or "ieee".

    Returns:
        Tensor of shape ``(..., n_out)``.
    """
    n_out, n_in = weight.shape
    if x.shape[-1] != n_in:
        raise ValueError(f"input width {x.shape[-1]} does not match weight {n_in}")

    x_flat = x.reshape(-1, n_in).contiguous()
    weight = weight.contiguous()
    out = torch.empty(x_flat.shape[0], n_out, device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(x_flat.shape[0], meta["BLOCK_M"]),
            triton.cdiv(n_out, meta["BLOCK_N"]),
        )

    linear_gelu_kernel[grid](
        x_flat,
        weight,
        bias,
        out,
        x_flat.shape[0],
        n_out,
        n_in,
        x_flat.stride(0),
        weight.stride(0),
        out.stride(0),
        PRECISION=precision,
    )
    return out.view(*x.shape[:-1], n_out)
