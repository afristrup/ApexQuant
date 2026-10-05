"""Scaled dot-product attention, streaming on CUDA and torch elsewhere."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .use_triton import use_triton


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Unmasked, dropout-free attention over ``(B, H, N, D)`` tensors.

    Args:
        q: Queries ``(B, H, N, D)``.
        k: Keys ``(B, H, N, D)``.
        v: Values ``(B, H, N, D)``.

    Returns:
        Attention output ``(B, H, N, D)``.
    """
    if use_triton(q):
        from ..kernels.triton_flash_attention import triton_flash_attention

        return triton_flash_attention(q, k, v)
    return F.scaled_dot_product_attention(q, k, v)
