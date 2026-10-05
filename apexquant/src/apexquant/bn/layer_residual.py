"""One quantized layer's weight error, with the group size that produced it."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LayerResidual:
    """``dW = W_hat - W`` for one layer, and how the quantizer grouped its row.

    The group size travels with the residual rather than being re-derived. It is
    a property of the quantization that happened, not of the weight shape: the
    harness picks the largest power of two at most 256 that *divides* the fan-in,
    while :func:`resolve_group_size` picks the largest power of two at most the
    configured group size, and the two disagree for most convolutions. Guessing
    would silently regroup the per-group ``tau`` of section 4.5, so it is carried.

    Attributes:
        dw: ``W_hat - W``, the same shape as the layer's weight, so the layout
            assertion of section 8.3 has something to check.
        group_size: Group size the quantizer used for this layer.
    """

    dw: torch.Tensor
    group_size: int
