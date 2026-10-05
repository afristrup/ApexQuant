"""apexquant.

ApexQuant is a data-free, calibration-free method for quantizing a weight
matrix to very low bit-widths. Each residual pass draws a fresh randomized
Hadamard rotation, L2-normalizes the rotated residual onto the unit sphere, and
rounds it with one of three interchangeable stages: scalar Lloyd-Max, an E8
lattice codebook, or a 16-state Viterbi-coded trellis. Because the rotation is
redrawn fresh every pass, the residual's marginal returns exactly to the
density each stage's codebook was built for, giving a closed-form geometric
decay law for the reconstruction error and letting every prefix of passes
serve as a valid lower-rate artifact ("nested decoding").

Quick start::

    import torch
    from apexquant import ApexQuantConfig, turboquantize_weight

    weight = torch.randn(4096, 4096)
    qw = turboquantize_weight(weight, ApexQuantConfig(bits=2, codebook="apex"))
    print(qw.nbytes())  # packed size in bytes
"""

from __future__ import annotations

from .analysis.error_tail_statistics import error_tail_statistics
from .analysis.layer_damage_table import layer_damage_table
from .analysis.linear_error_coefficients import linear_error_coefficients
from .analysis.random_token_divergence import random_token_divergence
from .codebook.apex_beta_levels import apex_beta_levels
from .codebook.generalized_gaussian_levels import generalized_gaussian_levels
from .codebook.lloyd_max_levels import lloyd_max_levels
from .codebook.lp_levels import lp_levels
from .codebook.uniform_levels import uniform_levels
from .config import ApexQuantConfig
from .fuse.knapsack_pass_allocation import knapsack_pass_allocation
from .fuse.sensitivity_residual_allocation import (
    measure_layer_sensitivity,
    sensitivity_residual_allocation,
)
from .fuse.validated_pass_allocation import validated_pass_allocation
from .kernels.fused_dequant_matmul import fused_dequant_matmul
from .nn.apex_linear import ApexQuantLinear
from .nn.linear_is_fusable import linear_is_fusable
from .nn.quantize_linear_modules import quantize_linear_modules
from .preprocess.sinkhorn_scales import sinkhorn_scales
from .quantize.dequantize_weight import dequantize_weight
from .quantize.quantized_weight import QuantizedWeight
from .quantize.resolve_group_size import resolve_group_size
from .quantize.turboquantize_weight import turboquantize_weight
from .throughput.benchmark_throughput import benchmark_throughput
from .throughput.cuda_graph_forward import cuda_graph_forward
from .throughput.memory_traffic_model import memory_traffic_model
from .vq.e8p_codec import E8PCodec
from .vq.progressive_artifact import dequantize as progressive_dequantize
from .vq.progressive_artifact import quantize as progressive_quantize
from .vq.trellis_codec import TrellisCodec

__version__ = "0.1.0"

__all__ = [
    "ApexQuantConfig",
    "ApexQuantLinear",
    "E8PCodec",
    "QuantizedWeight",
    "TrellisCodec",
    "apex_beta_levels",
    "benchmark_throughput",
    "cuda_graph_forward",
    "dequantize_weight",
    "error_tail_statistics",
    "fused_dequant_matmul",
    "generalized_gaussian_levels",
    "knapsack_pass_allocation",
    "layer_damage_table",
    "linear_error_coefficients",
    "linear_is_fusable",
    "lloyd_max_levels",
    "lp_levels",
    "measure_layer_sensitivity",
    "memory_traffic_model",
    "progressive_dequantize",
    "progressive_quantize",
    "quantize_linear_modules",
    "random_token_divergence",
    "resolve_group_size",
    "sensitivity_residual_allocation",
    "sinkhorn_scales",
    "turboquantize_weight",
    "uniform_levels",
    "validated_pass_allocation",
    "__version__",
]
