"""Apply a linear projection followed by GELU, whatever kind of layer it is."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .linear_gelu import linear_gelu


def gelu_projection(projection: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return ``gelu(projection(x))``, fused into one kernel where possible.

    :func:`linear_gelu` needs a dense weight to hand its kernel, which rules the
    fusion out for a quantized layer: those keep their weight packed, decode it
    inside their own matmul, and expose no ``weight`` at all. Rather than let the
    blocks reach past a layer for a tensor it may not have, they ask for the
    activation applied and this decides how.

    A packed layer may still fuse the activation into its own epilogue, and says
    so with ``fuses_gelu``. Taking that route matters: at Clay's expanding
    projection a separate GELU pass reads and writes 25 MB per layer.

    Args:
        projection: An ``nn.Linear``, or any module standing in for one.
        x: Input of shape ``(..., in_features)``.

    Returns:
        Tensor of shape ``(..., out_features)``.
    """
    if isinstance(projection, nn.Linear):
        return linear_gelu(x, projection.weight, projection.bias)
    if getattr(projection, "fuses_gelu", False):
        return projection(x, gelu=True)
    return F.gelu(projection(x))
