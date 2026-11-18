"""The no-rotation ablation applied to every target weight."""

from __future__ import annotations

import torch

from clinical.map_weights import map_weights

from .rtn_residual import rtn_residual


def quantize_residual_rtn(model, passes: int, group: int) -> dict[str, torch.Tensor]:
    """Two-bit residual RTN passes per target weight, ``group`` inputs per scale."""
    return map_weights(model, lambda w: rtn_residual(w, 2, passes, group))
