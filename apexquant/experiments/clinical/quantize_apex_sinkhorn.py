"""The ApexQuant trellis arm on SINQ's data-free Sinkhorn diagonals."""

from __future__ import annotations

import torch

from apexquant.preprocess.sinkhorn_scales import sinkhorn_scales

from .grouped_axis import grouped_axis
from .map_weights import map_weights
from .reconstruct_apex import reconstruct_apex
from .settings import GROUP

SINKHORN_ITERS = 10


def quantize_apex_sinkhorn(model, stages: list, seed: int) -> dict[str, torch.Tensor]:
    """Encode ``W / (r c)`` as the ApexQuant arm does and rescale by ``r c``.

    ``r`` and ``c`` come from the weight alone, so the arm stays data-free. In
    deployment ``r`` folds into the stored group scales and ``c`` into the input
    rotation, at the cost of one fp16 per row and column.
    """
    def one(weight: torch.Tensor) -> torch.Tensor:
        w = weight.float()
        row, col = sinkhorn_scales(w, SINKHORN_ITERS)
        outer = row[:, None] * col
        coded = grouped_axis(w / outer, lambda m: reconstruct_apex(m, stages, GROUP, seed),
                             GROUP)
        return (coded.float() * outer).to(weight.dtype)

    return map_weights(model, one)
