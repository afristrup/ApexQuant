"""Pre-norm multi-head self-attention block of the Clay transformer."""

from __future__ import annotations

import torch
from torch import nn

from ..ops.attention import attention
from ..ops.layer_norm import layer_norm


class SelfAttention(nn.Module):
    """Normalise, project to q/k/v, attend, project back.

    The residual add lives in :class:`~clay.model.transformer.Transformer`, not
    here. Parameter names match the original ``claymodel.backbone.Attention``.
    The original carried a ``fused_attn`` switch with a hand-written softmax
    fallback; every call site passed ``True``, so only the fused path survives
    and the choice of kernel now belongs to :func:`clay.ops.attention`.

    The two projections are called as modules, not as ``F.linear`` on their
    weights. Both are bias-free, so the two are the same arithmetic, but only
    the module call survives having the layer swapped for a quantized one.
    """

    def __init__(self, dim: int, heads: int, dim_head: int) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        x = layer_norm(x, self.norm.weight, self.norm.bias, self.norm.eps)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.view(batch, tokens, self.heads, self.dim_head).transpose(1, 2)
            for t in qkv
        )

        attended = attention(q, k, v)
        attended = attended.transpose(1, 2).reshape(batch, tokens, -1)
        return self.to_out(attended)
