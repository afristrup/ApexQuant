"""The ApexQuant trellis arm: data-free, one fixed stage, fresh frame per pass."""

from __future__ import annotations

import torch

from .grouped_axis import grouped_axis
from .map_weights import map_weights
from .reconstruct_apex import reconstruct_apex
from .settings import GROUP


def quantize_apex(model, stages: list, seed: int) -> dict[str, torch.Tensor]:
    """Reconstructed weight of every target weight, one pass per entry of ``stages``."""
    return map_weights(model, lambda w: grouped_axis(
        w, lambda m: reconstruct_apex(m, stages, GROUP, seed), GROUP))
