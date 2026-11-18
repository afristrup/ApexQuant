"""Every prefix of one ApexQuant encoding of one matrix."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from apexquant.rotation.inverse_random_rotation import inverse_random_rotation
from apexquant.rotation.random_rotation import random_rotation


@torch.no_grad()
def apex_prefixes(weight: torch.Tensor, stage, passes: int, group: int, seed: int
                  ) -> list[torch.Tensor]:
    """The reconstructions after one, two, ... ``passes`` passes of a single encoding.

    Same recursion and seeds as ``clinical.reconstruct_apex``, so entry ``P - 1``
    equals a ``P``-pass reconstruction exactly.
    """
    out_f, in_f = weight.shape
    pad = (-in_f) % group
    w = F.pad(weight.float(), (0, pad)).view(out_f, -1, group)
    residual, acc, prefixes = w, torch.zeros_like(w), []
    for p in range(passes):
        rotated = random_rotation(residual, seed + p)
        scale = rotated.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        coded = scale * stage((rotated / scale).reshape(-1, group)).view_as(rotated)
        back = inverse_random_rotation(coded, seed + p)
        acc, residual = acc + back, residual - back
        prefixes.append(acc.reshape(out_f, -1)[:, :in_f].to(weight.dtype))
    return prefixes
