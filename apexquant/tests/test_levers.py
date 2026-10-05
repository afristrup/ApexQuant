"""Tests for the improvement-lever primitives: E8 and the trellis.

``learned_rotation`` is dropped here: it is a calibration-requiring construction
that belongs to the excluded framework/spectral thread and was not ported.
"""

import torch

from apexquant.codebook import lloyd_max_levels, trellis_quantize
from apexquant.vq import e8_lattice_quantize


def test_e8_points_are_valid_lattice_points():
    torch.manual_seed(0)
    q = e8_lattice_quantize(torch.randn(1000, 8))
    # Every E8 point is in Z^8 or (Z+1/2)^8; doubling lands on integers either way.
    twice = 2 * q
    assert torch.allclose(twice, twice.round(), atol=1e-4)


def test_e8_beats_scalar_at_equal_rate():
    # Same rate (both unimodular at unit step); E8 has lower normalized 2nd moment.
    torch.manual_seed(0)
    x = (torch.rand(100000, 8) - 0.5) * 12
    mse_e8 = (x - e8_lattice_quantize(x)).pow(2).mean()
    mse_z8 = (x - x.round()).pow(2).mean()
    assert mse_e8 < mse_z8
    assert (1 - mse_e8 / mse_z8) > 0.10  # ~14% shaping gain


def test_trellis_beats_scalar_at_same_rate():
    g = torch.Generator().manual_seed(0)
    s = torch.randn(3000, generator=g)
    for r in (2, 3):
        levels = lloyd_max_levels(r)
        boundaries = 0.5 * (levels[1:] + levels[:-1])
        mse_scalar = (levels[torch.bucketize(s, boundaries)] - s).pow(2).mean()
        mse_trellis = (trellis_quantize(s, r) - s).pow(2).mean()
        assert mse_trellis < mse_scalar
