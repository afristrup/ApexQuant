"""Adapters for using ApexQuant artifacts inside vLLM model modules."""

from __future__ import annotations

import torch
from torch import nn

from ..inference import TrellisKVCache
from ..moe import ProgressiveE8MoE, ProgressiveMoE, ProgressiveScalarMoE, ProgressiveTrellisMoE
from ..nn.progressive_scalar_linear import ProgressiveScalarLinear


class ApexVLLMLinear(nn.Module):
    """Drop-in module for a vLLM linear with a scalar packed artifact."""

    def __init__(self, quantized_weight, bias=None):
        super().__init__()
        self.layer = ProgressiveScalarLinear(quantized_weight, bias)

    def forward(self, x):
        return self.layer(x)


def make_moe(stage: str, *args, **kwargs) -> ProgressiveMoE:
    """Construct the matching packed MoE adapter for ``scalar``, ``e8``, or ``trellis``."""
    classes = {"scalar": ProgressiveScalarMoE, "e8": ProgressiveE8MoE,
               "trellis": ProgressiveTrellisMoE}
    try:
        return classes[stage](*args, **kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown ApexQuant stage {stage!r}") from exc


def make_kv_cache(*args, **kwargs) -> TrellisKVCache:
    """Construct the ApexQuant trellis KV adapter for a vLLM model layer."""
    return TrellisKVCache(*args, **kwargs)
