"""Assert ApexQuant's core 2-bit claims reproduce (guards the replication study).

The two tests that went through ``QuantizationScheme``/``quantize_with_scheme``
are dropped: both live in the excluded ``framework``/``spectral`` thread and
were never ported. The density-matched-codebook claims below have no such
dependency and stand on their own.
"""

import torch

from apexquant.codebook import apex_beta_levels, uniform_levels
from apexquant.quantize import dequantize_from_indices, quantize_to_indices


def _marginal(d, n=60000, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, d, generator=g)
    return (v / v.norm(dim=1, keepdim=True))[:, 0]


def _mse(x, levels):
    return (dequantize_from_indices(quantize_to_indices(x, levels), levels) - x).pow(2).mean().item()


def test_codebook_gap_grows_with_fan_in():
    # Core claim: at 2 bits the density-matched codebook's edge over the uniform
    # grid widens with fan-in (Theta(log d)).
    gap = {}
    for d in (9, 512, 4096):
        x = _marginal(d)
        gap[d] = _mse(x, uniform_levels(2)) / _mse(x, apex_beta_levels(2, d))
    assert gap[9] < gap[512] < gap[4096]
    assert gap[9] < 5 and gap[4096] > 500


def test_low_fanin_boundary_degenerates_to_uniform():
    # Proposition 1: at d=3 the matched codebook IS the uniform grid.
    apex3 = apex_beta_levels(2, 3)
    assert torch.allclose(apex3, uniform_levels(2), atol=2e-2)
