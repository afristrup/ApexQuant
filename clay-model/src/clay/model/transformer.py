"""Stack of pre-norm attention and feed-forward blocks."""

from __future__ import annotations

import torch
from torch import nn

from ..ops.layer_norm import layer_norm
from .feed_forward import FeedForward
from .self_attention import SelfAttention


class Transformer(nn.Module):
    """The shared trunk of the Clay encoder and decoder.

    Layers are held as a ``ModuleList`` of two-element ``ModuleList``s rather
    than as a block module, which keeps the parameter keys (``layers.0.0`` for
    attention, ``layers.0.1`` for the feed-forward) identical to the original.
    """

    def __init__(  # noqa: PLR0913
        self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            nn.ModuleList(
                [
                    SelfAttention(dim, heads=heads, dim_head=dim_head),
                    FeedForward(dim, mlp_dim),
                ]
            )
            for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attend, feed_forward in self.layers:
            x = attend(x) + x
            x = feed_forward(x) + x
        return layer_norm(x, self.norm.weight, self.norm.bias, self.norm.eps)
