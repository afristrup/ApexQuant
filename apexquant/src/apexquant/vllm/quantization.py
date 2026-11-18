"""Out-of-tree vLLM quantization method backed by ApexQuant kernels."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from ..kernels.fused_dequant_matmul import fused_dequant_matmul
from ..quantize.quantized_weight import QuantizedWeight

try:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
except ImportError:  # pragma: no cover
    LinearBase = nn.Module
    RoutedExperts = nn.Module
    UnquantizedFusedMoEMethod = object
    UnquantizedLinearMethod = object
    QuantizationConfig = object


class ApexVLLMFusedMoEMethod(UnquantizedFusedMoEMethod):
    """Quantise fused ``(experts, output, input)`` weights before execution.

    vLLM has no packed-expert kernel here, so every stage writes its
    reconstruction back into the dense expert weights: the experts carry the
    stage's quality at FP16 memory and speed. Trellis and E8P experts are encoded
    ``APEXQUANT_MOE_ROW_BLOCK`` rows at a time (default 256).
    """

    def __init__(self, config, moe):
        super().__init__(moe)
        self.config = config

    def process_weights_after_loading(self, layer):
        from .. import ApexQuantConfig, dequantize_weight, turboquantize_weight
        from ..vq.progressive_artifact import dequantize, group_size_for, quantize

        quant_config = ApexQuantConfig(
            bits=self.config.bits,
            group_size=self.config.group_size,
            passes=self.config.passes,
            seed=self.config.seed,
            codebook="apex",
        )
        from .float32_default import float32_default

        row_block = int(os.environ.get("APEXQUANT_MOE_ROW_BLOCK", "256"))
        # APEXQUANT_SINKHORN_ITERS gives every expert its own data-free Sinkhorn
        # diagonals, as the trellis linears get theirs.
        iters = int(os.environ.get("APEXQUANT_SINKHORN_ITERS", "0"))
        if iters and self.config.stage == "scalar":
            raise NotImplementedError("Sinkhorn experts need a vector stage")
        from ..preprocess.sinkhorn_scales import sinkhorn_scales
        with torch.no_grad(), float32_default():
            for name in ("w13_weight", "w2_weight"):
                weight = getattr(layer, name)
                shape = weight.shape
                flat = weight.detach().reshape(-1, shape[-1]).float()
                if self.config.stage == "scalar":
                    restored = dequantize_weight(turboquantize_weight(flat, quant_config))
                else:
                    group_size = group_size_for(shape[-1], self.config.group_size)
                    codec = _stage_codec(self.config, group_size, flat.device)
                    experts = flat.view(shape[0], -1, shape[-1])
                    outer = None
                    if iters:
                        outer = torch.stack([
                            torch.outer(*sinkhorn_scales(e, iters)) for e in experts])
                        experts = experts / outer
                    flat_n = experts.reshape(-1, shape[-1])
                    codes, scale = quantize(flat_n, codec, self.config.passes, group_size,
                                            self.config.seed, row_block=row_block)
                    restored = dequantize(codes, scale, codec, shape[-1], group_size,
                                          self.config.seed, row_block=row_block)
                    if outer is not None:
                        restored = (restored.view_as(outer) * outer).reshape(flat.shape)
                weight.data.copy_(restored.reshape(shape).to(weight.dtype))
        super().process_weights_after_loading(layer)


def _stage_codec(config, group_size: int, device):
    """The trellis or E8P codec a vector stage encodes with."""
    if config.stage == "trellis":
        from ..vq.trellis_codec import TrellisCodec
        from .float32_default import float32_default

        with float32_default():
            return TrellisCodec(group_size, bits=config.bits,
                                trellis_bits=config.trellis_bits, seed=config.seed,
                                device=device)
    from .e8p_dispatch import e8p_codec

    return e8p_codec(group_size, str(device))


def _keep_dense(prefix: str) -> bool:
    """Routers, short convolutions and vision towers stay in FP16, as in E1.

    ``APEXQUANT_KEEP_DENSE`` is a regex of further prefixes to keep dense, for
    ablations such as leaving the early ``down_proj`` layers unquantized.
    """
    import re

    last = prefix.rsplit(".", 1)[-1]
    extra = os.environ.get("APEXQUANT_KEEP_DENSE")
    return (last in ("gate", "shared_expert_gate", "conv1d")
            or ".visual." in f".{prefix}" or "vision" in prefix
            or bool(extra and re.fullmatch(extra, prefix)))


class ApexVLLMLinearMethod(UnquantizedLinearMethod):
    """Runtime ApexQuant method for every vLLM linear layer."""

    def __init__(self, config):
        self.config = config

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ):
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        from .. import ApexQuantConfig, turboquantize_weight

        if getattr(layer, "_apexquant_processed", False):
            return
        weight = layer.weight.detach()
        quant_config = ApexQuantConfig(
            bits=self.config.bits,
            group_size=self.config.group_size,
            passes=self.config.passes,
            seed=self.config.seed,
            codebook="apex",
        )
        packed = turboquantize_weight(weight.float(), quant_config)
        if os.environ.get("APEXQUANT_DENSE_RECON") == "1":
            from .. import dequantize_weight

            dense = dequantize_weight(packed).to(weight.dtype)
            delattr(layer, "weight")
            layer.register_buffer("apex_dense", dense)
            layer._apexquant_processed = True
            return
        from ..kernels.fused_dequant_matmul import _stacked_rotations

        rotations = None
        if weight.is_cuda:
            rotations = _stacked_rotations(
                packed.group_size, packed.seed, packed.passes, str(weight.device)
            )
        for name, value in (
            ("apex_codes", packed.packed),
            ("apex_scale", packed.scale.to(torch.float16)),
            ("apex_levels", packed.levels),
            ("apex_row_scale", packed.row_scale),
            ("apex_col_scale", packed.col_scale),
            ("apex_rotations", rotations),
        ):
            layer.register_buffer(name, value)
        layer.apex_in_features = packed.in_features
        layer.apex_out_features = packed.out_features
        layer.apex_group_size = packed.group_size
        layer.apex_seed = packed.seed
        layer.apex_bits = packed.bits
        delattr(layer, "weight")
        layer._apexquant_processed = True

    def apply(self, layer: nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        dense = getattr(layer, "apex_dense", None)
        if dense is not None:
            return torch.nn.functional.linear(x.to(dense.dtype), dense, bias).to(x.dtype)
        qw = QuantizedWeight(
            packed=layer.apex_codes, scale=layer.apex_scale.float(),
            levels=layer.apex_levels, in_features=layer.apex_in_features,
            out_features=layer.apex_out_features,
            group_size=layer.apex_group_size, seed=layer.apex_seed,
            bits=layer.apex_bits, row_scale=layer.apex_row_scale,
            col_scale=layer.apex_col_scale, rotations=layer.apex_rotations,
            graph_safe=self.config.graph_safe)
        if qw.col_scale is not None:
            x = x * qw.col_scale.to(x.dtype)
        out = fused_dequant_matmul(x.float(), qw).to(x.dtype)
        if qw.row_scale is not None:
            out = out * qw.row_scale.to(out.dtype)
        return out if bias is None else out + bias


class ApexVLLMQuantizationConfig(QuantizationConfig):
    """vLLM config for the scalar packed ApexQuant format."""

    def __init__(
        self,
        bits: int = 2,
        group_size: int = 512,
        passes: int = 3,
        seed: int = 42,
        stage: str = "scalar",
        moe: bool = True,
        kv: bool = False,
        trellis_bits: int = 14,
        row_block: int | None = 16,
    ):
        super().__init__()
        passes = int(os.environ.get("APEXQUANT_PASSES", passes))
        stage = os.environ.get("APEXQUANT_STAGE", stage)
        row_block = int(os.environ.get("APEXQUANT_ROW_BLOCK", row_block or 0)) or None
        if stage not in ("scalar", "e8p", "trellis"):
            raise ValueError("stage must be scalar, e8p, or trellis")
        if stage == "trellis" and trellis_bits not in (12, 14):
            raise ValueError("trellis_bits must be 12 or 14")
        if bits not in (2, 4, 8):
            raise ValueError("ApexQuant vLLM integration supports 2, 4, or 8 bits")
        if group_size <= 0 or passes <= 0:
            raise ValueError("group_size and passes must be positive")
        from ..codebook.apex_beta_levels import apex_beta_levels
        self.bits = bits
        self.group_size = group_size
        self.passes = passes
        self.seed = seed
        self.stage = stage
        self.moe = moe
        self.kv = kv
        self.trellis_bits = trellis_bits
        self.row_block = row_block
        self.levels = apex_beta_levels(bits, group_size, device="cpu")
        self.graph_safe = True
        try:
            from vllm.config import get_current_vllm_config_or_none

            current = get_current_vllm_config_or_none()
            if current is not None:
                self.graph_safe = not current.model_config.enforce_eager
        except ImportError:
            pass

    def get_name(self) -> str:
        return "apexquant"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        return cls(
            bits=int(config.get("bits", 2)),
            group_size=int(config.get("group_size", 512)),
            passes=int(config.get("passes", 3)),
            seed=int(config.get("seed", 42)),
            stage=str(config.get("stage", "scalar")),
            moe=bool(config.get("moe", True)),
            kv=bool(config.get("kv", False)),
            trellis_bits=int(config.get("trellis_bits", 14)),
            row_block=config.get("row_block", 16),
        )

    def get_quant_method(self, layer: nn.Module, prefix: str):
        if isinstance(layer, RoutedExperts):
            if not self.moe:
                return None
            return ApexVLLMFusedMoEMethod(self, layer.moe_config)
        if isinstance(layer, LinearBase):
            if _keep_dense(prefix):
                return UnquantizedLinearMethod()
            if self.stage == "trellis":
                from .trellis_linear import TrellisLinearMethod
                return TrellisLinearMethod(self, prefix)
            if self.stage == "e8p":
                from .e8p_linear import E8PLinearMethod
                return E8PLinearMethod(self, prefix)
            return ApexVLLMLinearMethod(self)
        return None

    @staticmethod
    def get_supported_quantization() -> list[str]:
        return ["apexquant"]
