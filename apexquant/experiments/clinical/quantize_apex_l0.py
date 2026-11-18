"""The ApexQuant arm with decoder layer 0 kept in BF16."""

from __future__ import annotations

import torch

from .quantize_apex import quantize_apex
from .target_weights import target_weights


def quantize_apex_l0(model, stages: list, seed: int) -> dict[str, torch.Tensor]:
    """``quantize_apex`` on every target weight outside layer 0, which stays exact.

    On Llama-3.1-70B quantizing layer 0's attention collapses the model at four
    bits; keeping that one layer dense is a data-free rule.
    """
    weights = quantize_apex(model, stages, seed)
    for name, p in target_weights(model).items():
        if ".layers.0." in f".{name}":
            weights[name] = p.data.detach().cpu().clone()
    return weights
