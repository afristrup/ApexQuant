"""quadrature_moments against Monte Carlo, and the twelve-node accuracy claim."""

from __future__ import annotations

import math
import warnings

import pytest
import torch
import torch.nn.functional as F

from apexquant.bn.quadrature_moments import (DEFAULT_NODES,
                                                     quadrature_moments)
from apexquant.bn.relu_moments import relu_moments

SAMPLES = 4_000_000
SIGMA_TOLERANCES = 6.0
SEED = 20260914

#: Section 4.4's claim, which this file checks rather than assumes.
CLAIMED_NODES = 12
CLAIMED_RELATIVE_ERROR = 1e-5
#: A rule this fine is the reference the claim is measured against. Above about
#: 256 nodes numpy's hermgauss breaks down, which quadrature_moments rejects.
REFERENCE_NODES = 200
#: The band the sixteen-node default actually holds at unit sigma.
LOOSE_RELATIVE_ERROR = 1e-4
#: ReLU's kink is what Gauss-Hermite is worst at, so this cross-check of the
#: quadrature convention is deliberately coarse.
RELU_NODES = 128
RELU_RELATIVE_ERROR = 5e-3
#: A node count numpy's hermgauss cannot deliver.
BROKEN_NODES = 4096

ACTIVATIONS = {"silu": F.silu, "gelu": F.gelu}
#: Where the claim is measured: a wide sweep for Monte Carlo, and two narrower
#: sets for the node-count study.
CASES = [
    (0.0, 1.0),
    (2.0, 1.0),
    (-2.0, 1.0),
    (0.5, 3.0),
    (-1.0, 0.25),
    (1.0, 0.0),
]
NARROW_CASES = [(-1.0, 0.25), (0.0, 0.5), (2.0, 0.5)]
UNIT_SIGMA_CASES = [(0.0, 1.0), (2.0, 1.0), (-2.0, 1.0)]


def _relative_error(act, m, sigma, nodes):
    """Relative error of an ``nodes``-point rule against the reference rule."""
    mt = torch.tensor([m], dtype=torch.float64)
    st = torch.tensor([sigma], dtype=torch.float64)
    e_ref, s_ref = quadrature_moments(mt, st, act, nodes=REFERENCE_NODES)
    e, s = quadrature_moments(mt, st, act, nodes=nodes)
    return (abs(e.item() - e_ref.item()) / abs(e_ref.item()),
            abs(s.item() - s_ref.item()) / abs(s_ref.item()))


def _monte_carlo(m, sigma, act, seed):
    generator = torch.Generator().manual_seed(seed)
    z = m + sigma * torch.randn(SAMPLES, generator=generator, dtype=torch.float64)
    x = act(z)
    sq = x * x
    root_n = math.sqrt(SAMPLES)
    return (x.mean().item(), sq.mean().item(),
            x.std().item() / root_n, sq.std().item() / root_n)


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
@pytest.mark.parametrize("m,sigma", CASES)
def test_moments_match_monte_carlo(name, m, sigma):
    act = ACTIVATIONS[name]
    e, s = quadrature_moments(torch.tensor([m], dtype=torch.float64),
                              torch.tensor([sigma], dtype=torch.float64), act)
    if sigma == 0.0:
        assert e.item() == pytest.approx(act(torch.tensor(m, dtype=torch.float64)).item())
        return

    mc_e, mc_s, se_e, se_s = _monte_carlo(m, sigma, act, SEED + int(abs(m * 17) + sigma * 3))
    assert abs(e.item() - mc_e) < SIGMA_TOLERANCES * se_e, (e.item(), mc_e, se_e)
    assert abs(s.item() - mc_s) < SIGMA_TOLERANCES * se_s, (s.item(), mc_s, se_s)


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
@pytest.mark.parametrize("m,sigma", NARROW_CASES)
def test_twelve_nodes_reach_the_claimed_error_on_narrow_channels(name, m, sigma):
    """Section 4.4 claims twelve nodes give relative error below 1e-5 on both
    moments. It holds comfortably while sigma stays at or below half a unit,
    which is where the rule resolves the activation's transition region."""
    rel_e, rel_s = _relative_error(ACTIVATIONS[name], m, sigma, CLAIMED_NODES)

    assert rel_e < CLAIMED_RELATIVE_ERROR, (name, m, sigma, rel_e)
    assert rel_s < CLAIMED_RELATIVE_ERROR, (name, m, sigma, rel_s)


def test_twelve_nodes_miss_the_claimed_error_on_gelu_at_unit_sigma():
    """And the claim is false at the most ordinary operating point there is.
    GELU at m=0, sigma=1 has a twelve-node second moment off by roughly 1.2e-4,
    an order of magnitude past the stated 1e-5. This test pins the refutation so
    it cannot be quietly lost: the document overstates the rule's accuracy, and
    the sixteen-node default is doing real work rather than carrying margin."""
    rel_e, rel_s = _relative_error(F.gelu, 0.0, 1.0, CLAIMED_NODES)

    assert rel_e < CLAIMED_RELATIVE_ERROR, rel_e
    assert CLAIMED_RELATIVE_ERROR < rel_s < 1e-3, rel_s


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
@pytest.mark.parametrize("m,sigma", UNIT_SIGMA_CASES)
def test_sixteen_nodes_beat_twelve_at_unit_sigma(name, m, sigma):
    """The default is the smallest of the document's two node counts that keeps
    every unit-sigma case inside 1e-4 on both moments."""
    act = ACTIVATIONS[name]
    rel_e_12, rel_s_12 = _relative_error(act, m, sigma, CLAIMED_NODES)
    rel_e_16, rel_s_16 = _relative_error(act, m, sigma, DEFAULT_NODES)

    assert rel_e_16 <= rel_e_12 and rel_s_16 <= rel_s_12
    assert rel_e_16 < LOOSE_RELATIVE_ERROR and rel_s_16 < LOOSE_RELATIVE_ERROR


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
def test_wide_channels_need_far_more_than_the_default(name):
    """A channel three standard deviations wide is not resolved by any node count
    the document names: the transition region of SiLU and GELU occupies a
    vanishing share of the integration range and the rule converges slowly.
    Recorded so that a caller with large sigma knows to raise ``nodes``."""
    _, rel_s = _relative_error(ACTIVATIONS[name], 0.5, 3.0, DEFAULT_NODES)

    assert rel_s > LOOSE_RELATIVE_ERROR, (name, rel_s)


def test_default_is_sixteen_nodes_and_is_at_least_as_good_as_twelve():
    assert DEFAULT_NODES == 16
    mt = torch.tensor([0.5], dtype=torch.float64)
    st = torch.tensor([1.0], dtype=torch.float64)

    e_default, s_default = quadrature_moments(mt, st, F.silu)
    e_explicit, s_explicit = quadrature_moments(mt, st, F.silu, nodes=DEFAULT_NODES)
    e_ref, s_ref = quadrature_moments(mt, st, F.silu, nodes=REFERENCE_NODES)

    assert torch.equal(e_default, e_explicit) and torch.equal(s_default, s_explicit)
    assert abs(e_default.item() - e_ref.item()) < LOOSE_RELATIVE_ERROR * abs(e_ref.item())
    assert abs(s_default.item() - s_ref.item()) < LOOSE_RELATIVE_ERROR * abs(s_ref.item())


def test_an_identity_activation_recovers_the_gaussian_moments_exactly():
    """The rule is exact for polynomials up to degree 2n-1, so g(z) = z must give
    e = m and s = m^2 + sigma^2 to machine precision."""
    m = torch.tensor([-2.0, 0.0, 3.5], dtype=torch.float64)
    sigma = torch.tensor([0.0, 1.0, 2.25], dtype=torch.float64)

    e, s = quadrature_moments(m, sigma, lambda z: z)

    assert torch.allclose(e, m, atol=1e-12)
    assert torch.allclose(s, m * m + sigma * sigma, atol=1e-12)


def test_zero_sigma_needs_no_special_case():
    m = torch.tensor([-1.0, 2.0], dtype=torch.float64)
    sigma = torch.zeros(2, dtype=torch.float64)

    e, s = quadrature_moments(m, sigma, F.silu)

    assert torch.allclose(e, F.silu(m), atol=1e-12)
    assert torch.allclose(s, F.silu(m) ** 2, atol=1e-12)


def test_quadrature_agrees_with_the_closed_form_rectifier():
    """ReLU is not smooth, so the rule converges at a crawl here: 128 nodes buy
    only three digits. This is a cross-check of the quadrature convention
    (hermgauss integrates against exp(-t^2)), not of its convergence rate."""
    m = torch.tensor([0.7], dtype=torch.float64)
    sigma = torch.tensor([1.3], dtype=torch.float64)

    e_q, s_q = quadrature_moments(m, sigma, F.relu, nodes=RELU_NODES)
    e_c, s_c = relu_moments(m, sigma)

    assert e_q.item() == pytest.approx(e_c.item(), rel=RELU_RELATIVE_ERROR)
    assert s_q.item() == pytest.approx(s_c.item(), rel=RELU_RELATIVE_ERROR)


def test_broadcasting_and_shapes():
    m = torch.randn(4, 3, dtype=torch.float64)
    sigma = torch.rand(3, dtype=torch.float64)

    e, s = quadrature_moments(m, sigma, F.gelu)

    assert e.shape == (4, 3) and s.shape == (4, 3)


def test_bad_arguments_fail_fast():
    m = torch.zeros(2, dtype=torch.float64)
    sigma = torch.ones(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="at least 1"):
        quadrature_moments(m, sigma, F.silu, nodes=0)
    with pytest.raises(ValueError, match="non-negative"):
        quadrature_moments(m, -sigma, F.silu)
    with pytest.raises(ValueError, match="elementwise"):
        quadrature_moments(m, sigma, lambda z: z.sum(dim=-1))


def test_a_broken_quadrature_rule_fails_fast():
    """numpy's hermgauss overflows above a few hundred nodes and returns nan
    without raising, which would turn every moment into a silent nan."""
    m = torch.zeros(2, dtype=torch.float64)
    sigma = torch.ones(2, dtype=torch.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match=r"sqrt\(pi\)|non-finite"):
            quadrature_moments(m, sigma, F.silu, nodes=BROKEN_NODES)
