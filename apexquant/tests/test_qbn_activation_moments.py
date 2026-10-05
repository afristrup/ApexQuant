"""Monte Carlo agreement of the activation-moment dispatch of QBN section 4.4."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from apexquant.bn.activation_moments import (RELU6_HIGH, RELU6_LOW,
                                                    WIDE_NODES, activation_moments)
from apexquant.bn.activation_kinds import (ACT_GELU, ACT_HARDSIGMOID,
                                                  ACT_HARDSWISH, ACT_IDENTITY,
                                                  ACT_IMAGE, ACT_RELU,
                                                  ACT_RELU6, ACT_SIGMOID,
                                                  ACT_SILU, ACT_UNKNOWN)
from apexquant.bn.quadrature_moments import quadrature_moments

SAMPLES = 4_000_000

#: Loose on purpose. This bound is set by the two kinked activations, whose
#: Gauss-Hermite error plateaus near 2e-3 and does not improve with nodes, and by
#: the sampling error of four million draws. The typos it exists to catch are two
#: orders of magnitude larger: the Appendix C variance prints 0.261 against a true
#: 0.341, and Gast and Roth's ``(mu + v)`` for ``(mu^2 + v)`` is worse still.
#: The tight accuracy gate is
#: ``test_smooth_quadrature_kinds_converge_to_a_reference_rule``.
MC_TOLERANCE = 6e-3
CLOSED_FORM_TOLERANCE = 1e-6
REFERENCE_NODES = 256
REFERENCE_TOLERANCE = 1e-5

CASES = [
    (ACT_IDENTITY, lambda z: z),
    (ACT_RELU, torch.relu),
    (ACT_RELU6, lambda z: torch.clamp(z, RELU6_LOW, RELU6_HIGH)),
    (ACT_SILU, F.silu),
    (ACT_GELU, F.gelu),
    (ACT_SIGMOID, torch.sigmoid),
    (ACT_HARDSWISH, F.hardswish),
    (ACT_HARDSIGMOID, F.hardsigmoid),
]

GRID = [(0.0, 1.0), (1.5, 0.5), (-2.0, 2.0), (0.3, 0.05), (-0.7, 1.3)]


def _monte_carlo(g, m, sigma, seed):
    torch.manual_seed(seed)
    z = m + sigma * torch.randn(SAMPLES, dtype=torch.float64)
    x = g(z)
    return float(x.mean()), float((x * x).mean())


@pytest.mark.parametrize("kind,g", CASES)
@pytest.mark.parametrize("m,sigma", GRID)
def test_moments_match_monte_carlo(kind, g, m, sigma):
    """Every dispatched formula reproduces sampling, which is how the typos were caught."""
    e, s = activation_moments(kind, torch.tensor([m]), torch.tensor([sigma]))
    mc_e, mc_s = _monte_carlo(g, m, sigma, seed=hash((kind, m, sigma)) % 2**31)
    scale = max(1.0, abs(mc_e), abs(mc_s))
    assert abs(float(e) - mc_e) < MC_TOLERANCE * scale
    assert abs(float(s) - mc_s) < MC_TOLERANCE * scale


def test_identity_is_exact_algebra():
    """``identity`` is ``e = m`` and ``s = m^2 + sigma^2``, with no quadrature."""
    m = torch.tensor([-1.0, 0.0, 2.5])
    sigma = torch.tensor([0.0, 1.0, 3.0])
    e, s = activation_moments(ACT_IDENTITY, m, sigma)
    assert torch.allclose(e, m, atol=CLOSED_FORM_TOLERANCE)
    assert torch.allclose(s, m * m + sigma * sigma, atol=CLOSED_FORM_TOLERANCE)


def test_second_moment_is_raw_not_variance():
    """``s`` is ``E[x^2]``; a caller reading it as a variance would get ``tau`` wrong."""
    e, s = activation_moments(ACT_IDENTITY, torch.tensor([4.0]), torch.tensor([1.0]))
    assert float(s) == pytest.approx(17.0)
    assert float(s - e * e) == pytest.approx(1.0)


def test_relu_matches_the_closed_form_it_dispatches_to():
    """The dispatch adds nothing to ``relu_moments`` beyond the name lookup."""
    from apexquant.bn.relu_moments import relu_moments
    m = torch.linspace(-3.0, 3.0, 17)
    sigma = torch.linspace(0.0, 2.0, 17)
    dispatched = activation_moments(ACT_RELU, m, sigma)
    direct = relu_moments(m, sigma)
    assert torch.allclose(dispatched[0], direct[0])
    assert torch.allclose(dispatched[1], direct[1])


def test_relu6_is_the_clipped_normal_on_zero_six():
    """``relu6`` routes to the clipped-linear formula with the right interval."""
    from apexquant.bn.clipped_linear_moments import clipped_linear_moments
    m = torch.tensor([0.0, 3.0, 9.0])
    sigma = torch.tensor([1.0, 2.0, 0.5])
    dispatched = activation_moments(ACT_RELU6, m, sigma)
    direct = clipped_linear_moments(m, sigma, RELU6_LOW, RELU6_HIGH)
    assert torch.allclose(dispatched[0], direct[0])
    assert torch.allclose(dispatched[1], direct[1])


def test_smooth_quadrature_kinds_converge_to_a_reference_rule():
    """SiLU, GELU and Sigmoid are analytic, so the chosen rule must be converged.

    The two ``hard`` activations are excluded deliberately: their kinks put a
    floor of about 2e-3 on any Gauss-Hermite rule, exactly as section 4.4 warns
    for ReLU, and no node count removes it. Neither falsification model uses
    them -- ResNet-50 is all ReLU and EfficientNet-B0 is SiLU and Sigmoid -- so
    that floor is documented here rather than worked around.
    """
    for kind, g in [(ACT_SILU, F.silu), (ACT_GELU, F.gelu), (ACT_SIGMOID, torch.sigmoid)]:
        for m_value, sigma_value in GRID:
            m = torch.tensor([m_value], dtype=torch.float64)
            sigma = torch.tensor([sigma_value], dtype=torch.float64)
            e, s = activation_moments(kind, m, sigma)
            ref_e, ref_s = quadrature_moments(m, sigma, g, nodes=REFERENCE_NODES)
            assert abs(float(e - ref_e)) < REFERENCE_TOLERANCE, (kind, m_value, sigma_value)
            assert abs(float(s - ref_s)) < REFERENCE_TOLERANCE, (kind, m_value, sigma_value)


def test_wide_channels_get_the_larger_quadrature_rule():
    """Past one standard deviation, sixteen nodes are not enough and the rule grows."""
    m = torch.tensor([0.0])
    sigma = torch.tensor([3.0])
    got = activation_moments(ACT_SILU, m, sigma)
    reference = quadrature_moments(m, sigma, F.silu, nodes=200)
    coarse = quadrature_moments(m, sigma, F.silu, nodes=16)
    assert abs(float(got[1] - reference[1])) < abs(float(coarse[1] - reference[1]))
    assert WIDE_NODES > 16


def test_narrow_channels_keep_the_sixteen_node_default():
    """The default rule is load bearing below the threshold and is not silently raised."""
    m = torch.tensor([0.4])
    sigma = torch.tensor([0.5])
    got = activation_moments(ACT_SILU, m, sigma)
    default = quadrature_moments(m, sigma, F.silu, nodes=16)
    assert torch.equal(got[0], default[0])
    assert torch.equal(got[1], default[1])


def test_zero_sigma_is_deterministic_everywhere():
    """``gamma = 0`` is a real configuration and must not divide by zero."""
    m = torch.tensor([-1.0, 0.0, 2.0])
    sigma = torch.zeros(3)
    for kind, g in CASES:
        e, s = activation_moments(kind, m, sigma)
        assert torch.allclose(e, g(m), atol=1e-6), kind
        assert torch.allclose(s, g(m) ** 2, atol=1e-6), kind


@pytest.mark.parametrize("kind", [ACT_IMAGE, ACT_UNKNOWN, "not-an-activation"])
def test_unmodelled_activations_are_rejected(kind):
    """Cases C and D are uncorrected by construction and must never get moments."""
    with pytest.raises(ValueError):
        activation_moments(kind, torch.zeros(3), torch.ones(3))


def test_negative_sigma_is_rejected():
    """A negative sigma means the caller passed gamma where |gamma| belongs."""
    with pytest.raises(ValueError):
        activation_moments(ACT_RELU, torch.zeros(3), -torch.ones(3))
    with pytest.raises(ValueError):
        activation_moments(ACT_SILU, torch.zeros(3), -torch.ones(3))


def test_sigmoid_moments_are_bounded():
    """A sanity bound no sampling error can hide: sigmoid lives in (0, 1)."""
    m = torch.linspace(-6.0, 6.0, 25)
    sigma = torch.full((25,), 2.0)
    e, s = activation_moments(ACT_SIGMOID, m, sigma)
    assert bool(((e > 0.0) & (e < 1.0)).all())
    assert bool((s >= e * e - 1e-9).all())
    assert bool((s < 1.0).all())


def test_gelu_first_moment_matches_its_own_closed_form():
    """GELU's *first* moment has a closed form; the quadrature must reproduce it."""
    m, sigma = 0.75, 1.4
    exact = m * torch.special.ndtr(torch.tensor(m / math.sqrt(1.0 + sigma * sigma))) \
        + sigma * sigma / math.sqrt(2.0 * math.pi * (1.0 + sigma * sigma)) \
        * math.exp(-0.5 * m * m / (1.0 + sigma * sigma))
    e, _ = activation_moments(ACT_GELU, torch.tensor([m]), torch.tensor([sigma]))
    assert float(e) == pytest.approx(float(exact), abs=1e-4)
