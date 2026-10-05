"""Launch the streaming attention kernel over a ``(B, H, N, D)`` batch."""

from __future__ import annotations

import math

import torch
import triton

from ..constants import TRITON_DOT_PRECISION
from .flash_attention_kernel import LOG2_E, flash_attention_kernel


def triton_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    precision: str = TRITON_DOT_PRECISION,
) -> torch.Tensor:
    """Unmasked, dropout-free scaled dot-product attention.

    Args:
        q: Queries ``(B, H, N, D)``.
        k: Keys ``(B, H, N, D)``.
        v: Values ``(B, H, N, D)``.
        precision: ``tl.dot`` accumulation mode, "tf32" or "ieee".

    Returns:
        Attention output ``(B, H, N, D)``.
    """
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v shapes disagree: {q.shape}, {k.shape}, {v.shape}")

    batch, heads, n_tokens, head_dim = q.shape
    block_d = triton.next_power_of_2(head_dim)

    q_flat = q.reshape(batch * heads, n_tokens, head_dim).contiguous()
    k_flat = k.reshape(batch * heads, n_tokens, head_dim).contiguous()
    v_flat = v.reshape(batch * heads, n_tokens, head_dim).contiguous()
    out = torch.empty_like(q_flat)

    def grid(meta):
        return (triton.cdiv(n_tokens, meta["BLOCK_M"]), batch * heads)

    flash_attention_kernel[grid](
        q_flat,
        k_flat,
        v_flat,
        out,
        LOG2_E.value / math.sqrt(head_dim),
        q_flat.stride(0),
        q_flat.stride(1),
        n_tokens,
        head_dim,
        PRECISION=precision,
        EVEN_D=head_dim == block_d,
        BLOCK_D=block_d,
    )
    return out.view(batch, heads, n_tokens, head_dim)
