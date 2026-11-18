"""Apply a per-matrix quantizer to every target weight."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .target_weights import target_weights


@torch.no_grad()
def map_weights(model, fn: Callable[[torch.Tensor], torch.Tensor]) -> dict[str, torch.Tensor]:
    """``fn`` on each ``(out, in)`` matrix on the GPU, expert tensors slice by slice."""
    out = {}
    for name, p in target_weights(model).items():
        if p.dim() == 2:
            out[name] = fn(p.data.cuda()).cpu()
        else:
            out[name] = torch.stack([fn(m.cuda()).cpu() for m in p.data])
    return out
