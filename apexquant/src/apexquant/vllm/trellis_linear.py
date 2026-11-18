"""Dense vLLM linear method backed by packed Trellis weights."""

from __future__ import annotations

from typing import Any
import os

import torch
from torch import nn

from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from ..kernels.triton_gemv_trellis import trellis_window_table
from ..rotation.stacked_signs import stacked_signs
from .trellis_dispatch import trellis_linear, trellis_rotate
from ..vq.progressive_artifact import dequantize, group_size_for
from .cached_codes import cached_codes
from .float32_default import float32_default
from .trellis_rotate_scaled import trellis_rotate_scaled
from ..preprocess.sinkhorn_scales import sinkhorn_scales
from ..vq.trellis_codec import TrellisCodec


class TrellisLinearMethod(UnquantizedLinearMethod):
    """Load a dense vLLM linear and replace it with packed Trellis weights."""

    def __init__(self, config: Any, prefix: str = "") -> None:
        self.config = config
        self.prefix = prefix

    def create_weights(self, layer: nn.Module, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **attrs: Any) -> None:
        super().create_weights(
            layer, input_size_per_partition, output_partition_sizes, input_size,
            output_size, params_dtype, **attrs)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Encode the loaded weight; ``APEXQUANT_DENSE_RECON=1`` keeps its FP16 reconstruction.

        The reconstruction has the packed layer's quality at FP16 speed and
        memory, for accuracy runs only. ``APEXQUANT_TRELLIS_DENSE_CACHE`` used to
        keep the original weight under that name and so served FP16; it is refused.
        """
        if getattr(layer, "_apexquant_trellis_processed", False):
            return
        if "APEXQUANT_TRELLIS_DENSE_CACHE" in os.environ:
            raise RuntimeError("APEXQUANT_TRELLIS_DENSE_CACHE served unquantized weights; "
                               "use APEXQUANT_DENSE_RECON=1 for the reconstruction")
        weight_dtype = layer.weight.dtype
        weight = layer.weight.detach().float()
        # APEXQUANT_SINKHORN_ITERS > 0 encodes weight / (row x col), SINQ's
        # data-free diagonals. Rows fold into the per-group scales and columns into
        # the input rotation kernel, so serving pays for neither.
        iters = int(os.environ.get("APEXQUANT_SINKHORN_ITERS", "0"))
        if iters:
            row, col = sinkhorn_scales(weight, iters)
            weight = weight / (row[:, None] * col)
        # APEXQUANT_OUTLIER_Z > 0 keeps every weight beyond z standard deviations
        # exact (sparse FP16) and encodes the rest: one super weight would otherwise
        # dominate its group's norm and swamp the other 511 coordinates.
        z = float(os.environ.get("APEXQUANT_OUTLIER_Z", "0"))
        outliers = None
        if z:
            keep = weight.abs() > z * weight.std()
            outliers = torch.where(keep, weight, torch.zeros_like(weight))
            weight = weight - outliers
            print(f"apexquant outliers {self.prefix}: {int(keep.sum())} of {keep.numel()} "
                  f"beyond {z} sd", flush=True)
        group_size = group_size_for(weight.shape[1], self.config.group_size)
        with float32_default():
            codec = TrellisCodec(group_size, bits=self.config.bits,
                                 trellis_bits=self.config.trellis_bits,
                                 seed=self.config.seed, device=weight.device)
        tag = (f"{os.environ.get('APEXQUANT_CACHE_TAG', 'model')}"
               f"-b{self.config.bits}-L{self.config.trellis_bits}"
               + (f"-sinkhorn{iters}" if iters else "") + (f"-out{z:g}" if z else ""))
        codes, scale = cached_codes(self.prefix, "trellis", weight, codec,
                                    self.config.passes, group_size, self.config.seed,
                                    self.config.row_block, tag)
        layer.apex_trellis_out_features = weight.shape[0]
        delattr(layer, "weight")
        layer._apexquant_trellis_processed = True
        if os.environ.get("APEXQUANT_DENSE_RECON") == "1":
            with float32_default():
                dense = dequantize(codes, scale, codec, weight.shape[1], group_size,
                               self.config.seed)
            if iters:
                dense = dense * (row[:, None] * col)
            if outliers is not None:
                dense = dense + outliers
            layer.register_buffer("apex_trellis_dense", dense.to(weight_dtype))
            return
        if outliers is not None:
            raise NotImplementedError("packed trellis has no sparse outlier path; "
                                      "use APEXQUANT_DENSE_RECON=1")
        scale = scale.float()
        if iters:
            scale = scale * row[None, :, None]
            groups = scale.shape[-1]
            padded = torch.ones(groups * group_size, device=col.device)
            padded[:col.numel()] = col
            layer.register_buffer("apex_trellis_col", padded.contiguous())

        words = codes.reshape(codes.shape[0], codes.shape[1], -1).view(torch.int32)
        signs = stacked_signs(group_size, self.config.seed, self.config.passes,
                              str(weight.device))
        for name, value in (("apex_trellis_words", words),
                            ("apex_trellis_scale", scale.contiguous()),
                            ("apex_trellis_table", trellis_window_table(codec)),
                            ("apex_trellis_signs", signs)):
            layer.register_buffer(name, value)
        layer.apex_trellis_groups = scale.shape[-1]
        layer.apex_trellis_group_size = group_size

    def apply(self, layer: nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        dense = getattr(layer, "apex_trellis_dense", None)
        if dense is not None:
            out = torch.nn.functional.linear(x2.to(dense.dtype), dense)
            out = out.to(x.dtype).view(*lead, layer.apex_trellis_out_features)
            return out if bias is None else out + bias
        group_size = layer.apex_trellis_group_size
        pad = layer.apex_trellis_groups * group_size - x2.shape[-1]
        if pad:
            x2 = torch.nn.functional.pad(x2, (0, pad))
        col = getattr(layer, "apex_trellis_col", None)
        if col is not None:
            rotated = trellis_rotate_scaled(x2, col, layer.apex_trellis_signs, group_size)
        else:
            rotated = trellis_rotate(
                x2, layer.apex_trellis_signs, group_size,
                self.config.seed, self.config.passes,
            )
        args = (layer.apex_trellis_words, layer.apex_trellis_scale,
                layer.apex_trellis_table, group_size,
                self.config.bits, self.config.trellis_bits)
        out = trellis_linear(rotated, *args)
        out = out.to(x.dtype).view(*lead, layer.apex_trellis_out_features)
        return out if bias is None else out + bias
