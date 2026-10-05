"""Dispatch layer: each op picks its Triton kernel or its torch reference."""

from .attention import attention
from .gelu_projection import gelu_projection
from .gsd_frequency_ladder import gsd_frequency_ladder
from .layer_norm import layer_norm
from .linear_gelu import linear_gelu
from .sincos_1d import sincos_1d
from .sincos_2d_with_gsd import sincos_2d_with_gsd
from .use_triton import use_triton

__all__ = [
    "attention",
    "gelu_projection",
    "gsd_frequency_ladder",
    "layer_norm",
    "linear_gelu",
    "sincos_1d",
    "sincos_2d_with_gsd",
    "use_triton",
]
