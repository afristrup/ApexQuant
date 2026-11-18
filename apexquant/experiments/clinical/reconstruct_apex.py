"""ApexQuant residual passes on one weight matrix, reconstructed densely."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from apexquant.rotation.inverse_random_rotation import inverse_random_rotation
from apexquant.rotation.random_rotation import random_rotation


@torch.no_grad()
def reconstruct_apex(weight: torch.Tensor, stages: list, group: int,
                     seed: int) -> torch.Tensor:
    """One fresh-frame pass per entry of ``stages`` over groups of ``group`` inputs.

    Pass ``p`` uses rotation seed ``seed + p``, so the first ``P`` passes of a
    deeper encoding with the same leading stages are exactly this reconstruction:
    one artifact, read at ``P``.
    """
    out_f, in_f = weight.shape
    pad = (-in_f) % group
    w = F.pad(weight.float(), (0, pad)).view(out_f, -1, group)
    residual, acc = w, torch.zeros_like(w)
    for p, stage in enumerate(stages):
        rotated = random_rotation(residual, seed + p)
        scale = rotated.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        coded = scale * stage((rotated / scale).reshape(-1, group)).view_as(rotated)
        back = inverse_random_rotation(coded, seed + p)
        acc, residual = acc + back, residual - back
    return acc.reshape(out_f, -1)[:, :in_f].to(weight.dtype)
