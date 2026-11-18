"""HQQ, half-quadratic splitting, data-free."""

from __future__ import annotations

import torch

from .map_weights import map_weights


def quantize_hqq(model, bits: int, group: int) -> dict[str, torch.Tensor]:
    """HQQ's own dequantized weight for every target weight at its published axis."""
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

    cfg = BaseQuantizeConfig(nbits=bits, group_size=group, axis=1)

    def one(w: torch.Tensor) -> torch.Tensor:
        shell = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
        shell.weight.data = w.clone()
        hq = HQQLinear(shell, cfg, del_orig=True, compute_dtype=torch.bfloat16, device="cuda")
        return hq.dequantize().to(w.dtype)

    return map_weights(model, one)
