"""The stage's mean shrink factor, from the sphere alone."""

from __future__ import annotations

import torch


@torch.no_grad()
def stage_beta(stage, group: int, samples: int, device, seed: int = 0) -> float:
    """``beta = E(1 - <U, q(U)>)`` for ``U`` uniform on the unit sphere."""
    g = torch.Generator(device=device).manual_seed(seed)
    u = torch.randn(samples, group, device=device, generator=g)
    u = u / u.norm(dim=1, keepdim=True)
    return float(1 - (u * stage(u)).sum(1).mean())
