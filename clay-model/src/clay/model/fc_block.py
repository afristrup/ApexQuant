"""Residual two-layer GELU block over the wavelength embedding."""

from __future__ import annotations

import torch
from torch import nn

from ..ops.gelu_projection import gelu_projection


class FCBlock(nn.Module):
    """A residual ``gelu(W2 gelu(W1 x))`` refinement of the wave features.

    Submodule names match the original ``claymodel.factory.FCBlock`` so that
    published Clay checkpoints load without a key remap.
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(size, size)
        self.l2 = nn.Linear(size, size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = gelu_projection(self.l1, x)
        y = gelu_projection(self.l2, y)
        return x + y
