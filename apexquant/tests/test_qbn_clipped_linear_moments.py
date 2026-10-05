"""clipped_linear_moments against Monte Carlo, and against the ReLU limit."""

from __future__ import annotations

import math

import pytest
import torch

from apexquant.bn.clipped_linear_moments import clipped_linear_moments
from apexquant.bn.relu_moments import relu_moments

SAMPLES = 4_000_000
SIGMA_TOLERANCES = 6.0
SEED = 20260914

#: Section 4.4 quotes DFQ Appendix C at m=0, sigma=1 on this interval.
RELU6 = (0.0, 6.0)
PRINTED_VARIANCE = 0.261
TRUE_VARIANCE = 0.5 - 1.0 / (2.0 * math.pi)

CASES = [
    (0.0, 1.0, 0.0, 6.0),
    (3.0, 2.0, 0.0, 6.0),
    (-1.0, 0.5, 0.0, 6.0),
    (0.0, 4.0, -1.0, 1.0),
    (7.0, 1.0, 0.0, 6.0),
    (-4.0, 3.0, -2.0, 5.0),
    (0.25, 0.05, 0.0, 6.0),
]


def _monte_carlo(m, sigma, a, b, seed):
    """Return the sample moments of clip(N(m, sigma^2), a, b) and their errors."""
    generator = torch.Generator().manual_seed(seed)
    z = m + sigma * torch.randn(SAMPLES, generator=generator, dtype=torch.float64)
    x = torch.clamp(z, min=a, max=b)
    sq = x * x
    root_n = math.sqrt(SAMPLES)
    return (x.mean().item(), sq.mean().item(),
            x.std().item() / root_n, sq.std().item() / root_n)


@pytest.mark.parametrize("m,sigma,a,b", CASES)
def test_moments_match_monte_carlo(m, sigma, a, b):
    mc_e, mc_s, se_e, se_s = _monte_carlo(m, sigma, a, b, SEED + int(abs(m * 31) + b))
    e, s = clipped_linear_moments(torch.tensor([m], dtype=torch.float64),
                                  torch.tensor([sigma], dtype=torch.float64), a, b)

    assert abs(e.item() - mc_e) < SIGMA_TOLERANCES * se_e + 1e-9, (e.item(), mc_e, se_e)
    assert abs(s.item() - mc_s) < SIGMA_TOLERANCES * se_s + 1e-9, (s.item(), mc_s, se_s)


def test_the_two_published_numbers_on_the_documents_interval():
    m = torch.tensor([0.0], dtype=torch.float64)
    sigma = torch.tensor([1.0], dtype=torch.float64)
    e, s = clipped_linear_moments(m, sigma, *RELU6)
    variance = (s - e * e).item()

    assert variance == pytest.approx(0.341, abs=5e-4)
    # The upper clip at six sigma costs 2e-9 against the unbounded rectifier.
    assert variance == pytest.approx(TRUE_VARIANCE, abs=1e-8)
    assert abs(variance - PRINTED_VARIANCE) > 0.07, variance


@pytest.mark.parametrize("m,sigma", [(0.0, 1.0), (2.0, 1.0), (-2.0, 0.5)])
def test_a_far_upper_clip_reproduces_the_rectifier(m, sigma):
    """ReLU is the a=0, b=inf member of this family; at ten sigma above the mean
    the two must agree, which cross-checks the two independent derivations."""
    mt = torch.tensor([m], dtype=torch.float64)
    st = torch.tensor([sigma], dtype=torch.float64)
    far = m + 20.0 * max(sigma, 1.0)

    e_clip, s_clip = clipped_linear_moments(mt, st, 0.0, far)
    e_relu, s_relu = relu_moments(mt, st)

    assert torch.allclose(e_clip, e_relu, atol=1e-10)
    assert torch.allclose(s_clip, s_relu, atol=1e-10)


def test_zero_sigma_is_deterministic_and_does_not_divide_by_zero():
    m = torch.tensor([-5.0, 0.5, 9.0], dtype=torch.float64)
    sigma = torch.zeros(3, dtype=torch.float64)

    e, s = clipped_linear_moments(m, sigma, 0.0, 6.0)

    assert torch.allclose(e, torch.tensor([0.0, 0.5, 6.0], dtype=torch.float64))
    assert torch.allclose(s, torch.tensor([0.0, 0.25, 36.0], dtype=torch.float64))
    assert torch.isfinite(e).all() and torch.isfinite(s).all()


def test_second_moment_never_falls_below_the_square_of_the_first():
    """A variance that under-reports is exactly the failure the document warns
    about, and a negative one is its unmissable form."""
    torch.manual_seed(SEED)
    m = torch.randn(512, dtype=torch.float64) * 3.0
    sigma = torch.rand(512, dtype=torch.float64) * 4.0

    e, s = clipped_linear_moments(m, sigma, -2.0, 6.0)

    assert bool((s - e * e >= -1e-12).all())


def test_degenerate_or_infinite_levels_fail_fast():
    m = torch.zeros(2, dtype=torch.float64)
    sigma = torch.ones(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="finite"):
        clipped_linear_moments(m, sigma, 0.0, math.inf)
    with pytest.raises(ValueError, match="b > a"):
        clipped_linear_moments(m, sigma, 1.0, 1.0)


def test_negative_sigma_fails_fast():
    with pytest.raises(ValueError, match="non-negative"):
        clipped_linear_moments(torch.zeros(2), torch.tensor([1.0, -1.0]), 0.0, 6.0)
