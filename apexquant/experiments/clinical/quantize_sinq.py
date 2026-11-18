"""SINQ, Sinkhorn dual-scale, data-free, tiled at its published block size."""

from __future__ import annotations

import torch

from .map_weights import map_weights


def quantize_sinq(model, bits: int, group: int) -> dict[str, torch.Tensor]:
    """SINQ's fake-quantized weight for every target weight.

    ``sinq_fake_quant`` alone reduces over the whole fan-in; the tiled entry point
    is what the authors publish, so that is the one called here.
    """
    from sinq.awq import sinq_fake_quant, tiled_fake_quant_rectangle

    def one(w: torch.Tensor) -> torch.Tensor:
        if w.shape[1] % group:
            raise ValueError(f"fan-in {w.shape[1]} not divisible by {group}")
        recon = tiled_fake_quant_rectangle(
            w.float(), fakequant=sinq_fake_quant, min_max=[0, 2 ** bits - 1], block=group,
            scales=torch.ones(w.shape[1], device=w.device))
        return recon.to(w.dtype)

    return map_weights(model, one)
