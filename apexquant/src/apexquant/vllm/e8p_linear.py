"""Dense vLLM linear method backed by packed multi-pass E8P weights."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from ..rotation.stacked_signs import stacked_signs
from ..vq.progressive_artifact import dequantize, group_size_for
from .cached_codes import cached_codes
from .float32_default import float32_default
from .e8p_dispatch import e8p_codec, e8p_linear
from .trellis_dispatch import trellis_rotate


class E8PLinearMethod(UnquantizedLinearMethod):
    """Encode a loaded dense linear with E8P passes and serve the packed codes.

    ``APEXQUANT_DENSE_RECON=1`` serves the FP16 reconstruction of the same codes
    instead, which has the quality of the packed layer at the speed and memory of
    FP16; use it for accuracy only, never for throughput or memory.
    """

    def __init__(self, config: Any, prefix: str = "") -> None:
        self.config = config
        self.prefix = prefix

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if getattr(layer, "_apexquant_e8p_processed", False):
            return
        weight = layer.weight.detach().float()
        dtype = layer.weight.dtype
        k = weight.shape[1]
        group_size = group_size_for(k, self.config.group_size)
        codec = e8p_codec(group_size, str(weight.device))
        codes, scale = cached_codes(self.prefix, "e8p", weight, codec, self.config.passes,
                                    group_size, self.config.seed, self.config.row_block,
                                    os.environ.get("APEXQUANT_CACHE_TAG", "model"))
        delattr(layer, "weight")
        layer.apex_e8p_out_features = codes.shape[1]
        layer.apex_e8p_group_size = group_size
        layer._apexquant_e8p_processed = True
        if os.environ.get("APEXQUANT_DENSE_RECON") == "1":
            with float32_default():
                dense = dequantize(codes, scale, codec, k, group_size, self.config.seed)
            layer.register_buffer("apex_e8p_dense", dense.to(dtype))
            return
        flat = codes.reshape(codes.shape[0], codes.shape[1], -1).to(torch.int32)
        words = flat[..., 0::2] | (flat[..., 1::2] << 16)
        layer.register_buffer("apex_e8p_words", words.contiguous())
        layer.register_buffer("apex_e8p_scale", scale.float().contiguous())
        layer.register_buffer("apex_e8p_signs", stacked_signs(
            group_size, self.config.seed, self.config.passes, str(weight.device)))

    def apply(self, layer: nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        n = layer.apex_e8p_out_features
        dense = getattr(layer, "apex_e8p_dense", None)
        if dense is not None:
            out = torch.nn.functional.linear(x2.to(dense.dtype), dense).to(x.dtype)
            return (out if bias is None else out + bias).view(*lead, n)
        group_size = layer.apex_e8p_group_size
        pad = layer.apex_e8p_scale.shape[-1] * group_size - x2.shape[-1]
        if pad:
            x2 = torch.nn.functional.pad(x2, (0, pad))
        rotated = trellis_rotate(x2, layer.apex_e8p_signs, group_size, self.config.seed,
                                 self.config.passes)
        out = e8p_linear(rotated.float(), layer.apex_e8p_words, layer.apex_e8p_scale,
                         group_size).to(x.dtype).view(*lead, n)
        return out if bias is None else out + bias
