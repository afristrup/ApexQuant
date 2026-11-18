"""Replace every eligible encoder linear by its trellis reconstruction at ``P`` passes."""

from __future__ import annotations

import copy

import torch
from torch import nn

from apexquant.vq.progressive_artifact import dequantize, group_size_for, quantize
from apexquant.preprocess.sinkhorn_scales import sinkhorn_scales
from apexquant.vq.e8p_codec import E8PCodec
from apexquant.vq.scalar_codec import ScalarCodec
from apexquant.vq.trellis_codec import TrellisCodec

from .settings import GROUP, MIN_IN_FEATURES, SINKHORN_ITERS


@torch.no_grad()
def quantize_encoder(encoder: nn.Module, passes: int, stage: str = "trellis",
                     seed: int = 42) -> tuple[nn.Module, int]:
    """A copy whose linears (fan-in >= MIN_IN_FEATURES, outside nn.MultiheadAttention)
    carry the ``passes``-pass reconstruction of ``stage`` (trellis, e8p, scalar or
    trellis_sinkhorn, the last on SINQ's data-free diagonals); returns it and the count."""
    model = copy.deepcopy(encoder)
    fused = {id(c) for m in model.modules() if isinstance(m, nn.MultiheadAttention)
             for c in m.modules()}
    count = 0
    for module in model.modules():
        if (isinstance(module, nn.Linear) and module.in_features >= MIN_IN_FEATURES
                and id(module) not in fused):
            w = module.weight.float()
            outer = None
            if stage == "trellis_sinkhorn":
                outer = torch.outer(*sinkhorn_scales(w, SINKHORN_ITERS))
                w = w / outer
            gs = group_size_for(w.shape[1], GROUP)
            codec = {"e8p": lambda: E8PCodec(gs, seed=0, device=w.device),
                     "scalar": lambda: ScalarCodec(gs, bits=2, device=w.device)}.get(
                         stage, lambda: TrellisCodec(gs, bits=2, seed=seed, device=w.device))()
            codes, scale = quantize(w, codec, passes, gs, seed, row_block=256)
            r = dequantize(codes, scale, codec, w.shape[1], gs, seed)
            module.weight.copy_((r if outer is None else r * outer).to(module.weight.dtype))
            count += 1
    return model, count
