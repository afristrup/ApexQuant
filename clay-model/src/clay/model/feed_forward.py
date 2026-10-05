"""Pre-norm feed-forward block of the Clay transformer."""

from __future__ import annotations

import torch
from torch import nn

from ..ops.gelu_projection import gelu_projection
from ..ops.layer_norm import layer_norm


class FeedForward(nn.Module):
    """Normalise, expand through GELU, project back.

    The ``nn.Sequential`` is kept solely so the parameter keys remain ``net.0``,
    ``net.1`` and ``net.3`` and published Clay checkpoints load without a remap.
    The ``nn.GELU`` at index 2 holds no parameters and is never called: the
    activation is fused into the expanding projection by :func:`clay.ops.linear_gelu`.

    That fusion needs a dense weight to hand the kernel, so it applies only while
    the expanding projection is an ``nn.Linear``. Swap in a quantized layer, which
    keeps its weight packed and exposes no ``weight``, and the block calls it as a
    module and applies the activation separately.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm, expand, _, project = self.net
        x = layer_norm(x, norm.weight, norm.bias, norm.eps)
        x = gelu_projection(expand, x)
        return project(x)
