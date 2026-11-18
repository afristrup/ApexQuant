"""The scalar Lloyd-Max stage of the spherical marginal, as a callable."""

from __future__ import annotations

from apexquant.codebook.apex_beta_levels import apex_beta_levels
from apexquant.quantize.dequantize_from_indices import dequantize_from_indices
from apexquant.quantize.quantize_to_indices import quantize_to_indices


def scalar_stage(bits: int, group: int, device):
    """``u -> q(u)`` rounding each coordinate of a unit vector to the ``f_d`` codebook."""
    levels = apex_beta_levels(bits, group).to(device)
    return lambda u: dequantize_from_indices(quantize_to_indices(u, levels), levels)
