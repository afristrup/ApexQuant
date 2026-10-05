"""Fused dequant+matmul kernels (torch reference + Triton fast-paths)."""

from .fused_dequant_matmul import fused_dequant_matmul
from .torch_fused_dequant_matmul import torch_fused_dequant_matmul
from .triton_fused_dequant_matmul import TRITON_AVAILABLE, triton_fused_dequant_matmul

__all__ = [
    "TRITON_AVAILABLE",
    "fused_dequant_matmul",
    "torch_fused_dequant_matmul",
    "triton_fused_dequant_matmul",
]
