"""Neural-network modules: fused quantized layers and model conversion."""

from .quantize_linear_modules import quantize_linear_modules
from .apex_linear import ApexQuantLinear

__all__ = ["ApexQuantLinear", "quantize_linear_modules"]
