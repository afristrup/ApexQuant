"""relu_moments against Monte Carlo, and against the two published typos."""

from __future__ import annotations

import math

import pytest
import torch

from apexquant.bn.relu_moments import relu_moments

SAMPLES = 4_000_000
SIGMA_TOLERANCES = 6.0
SEED = 20260914
SQRT_TWO_PI = math.sqrt(2.0 * math.pi)

#: Section 4.4: DFQ Appendix C's printed variance gives this at m=0, sigma=1.
PRINTED_VARIANCE = 0.261
#: ... against this truth, which is 1/2 - 1/(2 pi).
TRUE_VARIANCE = 0.5 - 1.0 / (2.0 * math.pi)

CASES = [
    (0.0, 1.0),
    (2.0, 1.0),
    (-2.0, 1.0),
    (-0.5, 0.25),
    (1.0, 0.01),
    (-3.0, 3.0),
    (0.3, 7.0),
]


def _monte_carlo(m: float, sigma: float, seed: int) -> tuple[float, float, float, float]:
    """Return the sample moments of ReLU(N(m, sigma^2)) and their standard errors."""
    generator = torch.Generator().manual_seed(seed)
    z = m + sigma * torch.randn(SAMPLES, generator=generator, dtype=torch.float64)
    x = torch.clamp(z, min=0.0)
    sq = x * x
    root_n = math.sqrt(SAMPLES)
    return (x.mean().item(), sq.mean().item(),
            x.std().item() / root_n, sq.std().item() / root_n)


@pytest.mark.parametrize("m,sigma", CASES)
def test_moments_match_monte_carlo(m, sigma):
    mc_e, mc_s, se_e, se_s = _monte_carlo(m, sigma, SEED + int(abs(m * 100) + sigma * 10))
    e, s = relu_moments(torch.tensor([m], dtype=torch.float64),
                        torch.tensor([sigma], dtype=torch.float64))

    assert abs(e.item() - mc_e) < SIGMA_TOLERANCES * se_e + 1e-9, (e.item(), mc_e, se_e)
    assert abs(s.item() - mc_s) < SIGMA_TOLERANCES * se_s + 1e-9, (s.item(), mc_s, se_s)


def test_the_two_published_numbers():
    """0.261 is what the printed formula gives; 0.341 is the truth."""
    m = torch.tensor([0.0], dtype=torch.float64)
    sigma = torch.tensor([1.0], dtype=torch.float64)
    e, s = relu_moments(m, sigma)
    variance = (s - e * e).item()

    assert variance == pytest.approx(TRUE_VARIANCE, abs=1e-12)
    assert variance == pytest.approx(0.341, abs=5e-4)
    assert abs(variance - PRINTED_VARIANCE) > 0.07, variance

    mc_e, mc_s, se_e, se_s = _monte_carlo(0.0, 1.0, SEED)
    mc_variance = mc_s - mc_e * mc_e
    assert abs(mc_variance - TRUE_VARIANCE) < SIGMA_TOLERANCES * (se_s + 2 * se_e)
    assert abs(mc_variance - PRINTED_VARIANCE) > 0.07, mc_variance


@pytest.mark.parametrize("m,sigma", [(2.0, 1.0), (-1.0, 2.0), (0.5, 0.5)])
def test_the_gast_and_roth_typo_is_not_what_we_compute(m, sigma):
    """Eq. 10b prints (mu + v) where it must be (mu^2 + v)."""
    r = m / sigma
    cdf = 0.5 * (1.0 + math.erf(r / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * r * r) / SQRT_TWO_PI
    typo_s = (m + sigma * sigma) * cdf + m * sigma * pdf

    _, s = relu_moments(torch.tensor([m], dtype=torch.float64),
                        torch.tensor([sigma], dtype=torch.float64))
    _, mc_s, _, se_s = _monte_carlo(m, sigma, SEED + 7)

    assert abs(s.item() - mc_s) < SIGMA_TOLERANCES * se_s
    assert abs(typo_s - mc_s) > 10.0 * se_s, (typo_s, mc_s)


def test_zero_sigma_is_deterministic_and_does_not_divide_by_zero():
    m = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64)
    sigma = torch.zeros(3, dtype=torch.float64)

    e, s = relu_moments(m, sigma)

    assert torch.allclose(e, torch.tensor([0.0, 0.0, 3.0], dtype=torch.float64))
    assert torch.allclose(s, torch.tensor([0.0, 0.0, 9.0], dtype=torch.float64))
    assert torch.isfinite(e).all() and torch.isfinite(s).all()


def test_mixed_zero_and_positive_sigma_broadcasts():
    m = torch.tensor([1.0, 1.0], dtype=torch.float64)
    sigma = torch.tensor([0.0, 1.0], dtype=torch.float64)

    e, s = relu_moments(m, sigma)

    assert e[0].item() == pytest.approx(1.0)
    assert s[0].item() == pytest.approx(1.0)
    assert e[1].item() > 1.0


def test_negative_sigma_fails_fast():
    with pytest.raises(ValueError, match="non-negative"):
        relu_moments(torch.zeros(2), torch.tensor([1.0, -1.0]))
