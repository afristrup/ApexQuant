"""The DASHQuant arm: data-free, one randomized Hadamard transform, one pass."""

from __future__ import annotations

import torch

from .grouped_axis import grouped_axis
from .map_weights import map_weights
from .reconstruct_dash import reconstruct_dash
from .settings import GROUP


def quantize_dash(model, bits: int, seed: int) -> dict[str, torch.Tensor]:
    """Reconstructed weight of every target weight at ``bits`` per coordinate."""
    return map_weights(model, lambda w: grouped_axis(
        w, lambda m: reconstruct_dash(m, bits, GROUP, seed), GROUP))
