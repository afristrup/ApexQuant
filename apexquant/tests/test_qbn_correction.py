"""The per-channel correction of QBN section 4.5, against Monte Carlo."""

from __future__ import annotations

import pytest
import torch

from apexquant.bn.qbn_correction import qbn_correction
from apexquant.bn.qbn_guards import MAX_MEAN_SHIFT, VAR_FLOOR_RATIO
from apexquant.bn.row_alpha import row_alpha

OUT_FEATURES = 12
CHANNELS = 16
SPATIAL = 4
COLUMNS = CHANNELS * SPATIAL
GROUP_SIZE = 16
GROUPS = COLUMNS // GROUP_SIZE
SAMPLES = 600_000

#: The cross term vanishes only in expectation. These two tests construct cases
#: where it vanishes *identically*, so what is left is sampling error alone: a
#: variance estimate from ``SAMPLES`` draws carries a relative standard error of
#: ``sqrt(2 / SAMPLES)``, about 0.2%, and the tolerance is a few multiples of the
#: layer variance times that.
VAR_TOLERANCE = 5e-3
MEAN_TOLERANCE = 1e-2


def _column_moments(e: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand per-channel moments over the ``SPATIAL`` columns each channel owns."""
    return e.repeat_interleave(SPATIAL), s.repeat_interleave(SPATIAL)


def _sample_and_measure(w, dw, column_mean, column_sd, seed):
    """Return the exact stored statistics and the measured post-quantization ones."""
    generator = torch.Generator().manual_seed(seed)
    x = column_mean + column_sd * torch.randn(SAMPLES, COLUMNS, generator=generator)
    y_fp32 = x @ w.t()
    y_quant = x @ (w + dw).t()
    return (y_fp32.mean(dim=0), y_fp32.var(dim=0),
            y_quant.mean(dim=0), y_quant.var(dim=0))


def test_isotropic_covariance_reproduces_monte_carlo_exactly():
    """With ``Cov(x) = tau I`` the dropped cross term is identically zero.

    ``dWperp_c`` is orthogonal to ``W_c`` by construction, so
    ``W_c^T C dWperp_c = tau <W_c, dWperp_c> = 0``. Everything section 4.5 keeps
    is then the whole truth, and the prediction must match sampling to the noise
    floor. This is the check that the signs, the factor of two in
    ``2 alpha + alpha^2``, and the choice of ``tr(Cov x)/d`` over
    ``tr(E[x x^T])/d`` are all right.
    """
    torch.manual_seed(7)
    w = torch.randn(OUT_FEATURES, COLUMNS)
    dw = 0.12 * torch.randn(OUT_FEATURES, COLUMNS)
    tau = 0.37
    e = torch.randn(CHANNELS) * 0.5
    s = e * e + tau
    column_mean, _ = _column_moments(e, s)
    mu, v, measured_mean, measured_var = _sample_and_measure(
        w, dw, column_mean, tau ** 0.5, seed=101)

    correction = qbn_correction(w, dw, e, s, mu, v, GROUP_SIZE)

    assert bool(((correction.d_mean - (measured_mean - mu)).abs()
                 < MEAN_TOLERANCE * v.sqrt()).all())
    assert bool(((correction.d_var - (measured_var - v)).abs()
                 < VAR_TOLERANCE * v).all())
    assert bool((correction.tau - tau).abs().max() < 1e-5)


def test_per_group_tau_reproduces_monte_carlo_when_groups_differ():
    """The per-group decomposition of section 4.5 is exact, not an approximation.

    Each group gets its own input variance and ``dWperp`` is made orthogonal to
    ``W`` *within every group*, which again kills the cross term identically.
    A whole-row ``tau`` would be wrong here, so this also shows the per-group
    form is doing real work.
    """
    torch.manual_seed(8)
    w = torch.randn(OUT_FEATURES, COLUMNS)
    alpha_target = -0.014
    perp = torch.randn(OUT_FEATURES, GROUPS, GROUP_SIZE) * 0.1
    grouped_w = w.view(OUT_FEATURES, GROUPS, GROUP_SIZE)
    projection = (perp * grouped_w).sum(dim=2, keepdim=True) \
        / grouped_w.pow(2).sum(dim=2, keepdim=True)
    perp = perp - projection * grouped_w
    dw = (alpha_target * grouped_w + perp).reshape(OUT_FEATURES, COLUMNS)

    channel_var = torch.linspace(0.1, 2.0, CHANNELS)
    e = torch.randn(CHANNELS) * 0.3
    s = e * e + channel_var
    column_mean, column_second = _column_moments(e, s)
    column_sd = (column_second - column_mean * column_mean).sqrt()
    mu, v, measured_mean, measured_var = _sample_and_measure(
        w, dw, column_mean, column_sd, seed=202)

    correction = qbn_correction(w, dw, e, s, mu, v, GROUP_SIZE)

    assert float((correction.alpha - alpha_target).abs().max()) < 1e-5
    assert bool(((correction.d_mean - (measured_mean - mu)).abs()
                 < MEAN_TOLERANCE * v.sqrt()).all())
    assert bool(((correction.d_var - (measured_var - v)).abs()
                 < VAR_TOLERANCE * v).all())


def test_using_the_second_moment_as_tau_would_inflate_the_correction():
    """``tau`` is ``tr(Cov x)/d``; feeding ``tr(E[x x^T])/d`` double counts the mean.

    The document measures the damage: ResNet-50's median ``tau`` goes from 0.610
    to 0.955. Here the same mistake is made deliberately, by handing in a second
    moment whose mean has already been folded in, and the resulting ``tau`` is
    strictly larger everywhere the mean is nonzero.
    """
    torch.manual_seed(9)
    w = torch.randn(OUT_FEATURES, COLUMNS)
    dw = 0.1 * torch.randn(OUT_FEATURES, COLUMNS)
    e = torch.full((CHANNELS,), 0.8)
    s = e * e + 0.5
    mu = torch.zeros(OUT_FEATURES)
    v = torch.full((OUT_FEATURES,), 3.0)
    honest = qbn_correction(w, dw, e, s, mu, v, GROUP_SIZE)
    inflated = qbn_correction(w, dw, torch.zeros(CHANNELS), s, mu, v, GROUP_SIZE)
    assert float(honest.tau.median()) == pytest.approx(0.5, abs=1e-5)
    assert float(inflated.tau.median()) == pytest.approx(1.14, abs=1e-5)
    assert bool((inflated.d_var > honest.d_var).all())


def test_alpha_is_the_realized_projection_not_the_closed_form():
    """Section 8.1: the closed form must never be substituted for the measurement."""
    torch.manual_seed(10)
    w = torch.randn(OUT_FEATURES, COLUMNS)
    dw = 0.2 * torch.randn(OUT_FEATURES, COLUMNS)
    correction = qbn_correction(w, dw, torch.zeros(CHANNELS),
                                torch.ones(CHANNELS), torch.zeros(OUT_FEATURES),
                                torch.ones(OUT_FEATURES), GROUP_SIZE)
    assert torch.allclose(correction.alpha, row_alpha(w, dw))


def test_contraction_term_alone_over_corrects():
    """The two variance terms partly cancel, so they ship together or not at all."""
    torch.manual_seed(11)
    w = torch.randn(OUT_FEATURES, COLUMNS)
    alpha_target = -0.0137
    dw = alpha_target * w
    e = torch.zeros(CHANNELS)
    s = torch.ones(CHANNELS)
    v = torch.full((OUT_FEATURES,), 2.0)
    contraction_only = qbn_correction(w, dw, e, s, torch.zeros(OUT_FEATURES), v, GROUP_SIZE)
    assert bool((contraction_only.d_var < 0.0).all())
    expected = (2.0 * alpha_target + alpha_target ** 2) * 2.0
    assert float(contraction_only.d_var.median()) == pytest.approx(expected, rel=1e-4)


def test_variance_floor_guard_fires_and_is_counted():
    """``running_var`` can never fall below a quarter of its stored value."""
    w = torch.eye(OUT_FEATURES, COLUMNS)
    dw = -0.99 * w
    e = torch.zeros(CHANNELS)
    s = torch.ones(CHANNELS) * 1e-6
    v = torch.full((OUT_FEATURES,), 1.0)
    correction = qbn_correction(w, dw, e, s, torch.zeros(OUT_FEATURES), v, GROUP_SIZE)
    assert correction.var_floor_hits == OUT_FEATURES
    assert torch.allclose(correction.running_var, VAR_FLOOR_RATIO * v)
    assert bool((correction.d_var < -0.75).all())


def test_mean_shift_guard_clamps_and_is_counted():
    """A mean shift past three standard deviations is a case-assignment bug, not a shift."""
    w = torch.ones(OUT_FEATURES, COLUMNS)
    dw = torch.ones(OUT_FEATURES, COLUMNS)
    e = torch.full((CHANNELS,), 10.0)
    s = e * e + 1.0
    v = torch.full((OUT_FEATURES,), 1.0)
    mu = torch.zeros(OUT_FEATURES)
    correction = qbn_correction(w, dw, e, s, mu, v, GROUP_SIZE)
    assert correction.mean_shift_hits == OUT_FEATURES
    assert torch.allclose(correction.running_mean,
                          torch.full((OUT_FEATURES,), MAX_MEAN_SHIFT))
    assert float(correction.d_mean.median()) == pytest.approx(COLUMNS * 10.0)


def test_guards_are_inactive_at_realistic_magnitudes():
    """At ``P=2`` neither guard should fire; section 4.7 says so and it is checked."""
    torch.manual_seed(12)
    w = torch.randn(OUT_FEATURES, COLUMNS)
    dw = 0.117 * torch.randn(OUT_FEATURES, COLUMNS)
    e = torch.rand(CHANNELS)
    s = e * e + 0.6
    column_mean, _ = _column_moments(e, s)
    mu, v, _, _ = _sample_and_measure(w, dw, column_mean, 0.6 ** 0.5, seed=303)
    correction = qbn_correction(w, dw, e, s, mu, v, GROUP_SIZE)
    assert correction.mean_shift_hits == 0
    assert correction.var_floor_hits == 0


def test_padded_fan_in_is_rejected_rather_than_guessed():
    """The document fixes no convention for what channel a pad column names."""
    w = torch.randn(OUT_FEATURES, COLUMNS)
    with pytest.raises(ValueError, match="whole number of groups"):
        qbn_correction(w, w, torch.zeros(CHANNELS), torch.ones(CHANNELS),
                       torch.zeros(OUT_FEATURES), torch.ones(OUT_FEATURES), 24)


@pytest.mark.parametrize("mutate", [
    lambda kw: kw.update(dw=kw["dw"][:-1]),
    lambda kw: kw.update(w=kw["w"].reshape(OUT_FEATURES, CHANNELS, SPATIAL)),
    lambda kw: kw.update(mu=kw["mu"][:-1]),
    lambda kw: kw.update(v=kw["v"][:-1]),
    lambda kw: kw.update(s=kw["s"][:-1]),
    lambda kw: kw.update(e=kw["e"][:-1]),
])
def test_shape_disagreements_fail_fast(mutate):
    """Every shape the correction relies on is checked, not trusted."""
    kw = dict(w=torch.randn(OUT_FEATURES, COLUMNS), dw=torch.randn(OUT_FEATURES, COLUMNS),
              e=torch.zeros(CHANNELS), s=torch.ones(CHANNELS),
              mu=torch.zeros(OUT_FEATURES), v=torch.ones(OUT_FEATURES),
              group_size=GROUP_SIZE)
    mutate(kw)
    with pytest.raises(ValueError):
        qbn_correction(**kw)


def test_zero_error_is_a_zero_correction():
    """A dense layer gets ``dW = 0``, and the buffers must come back untouched."""
    torch.manual_seed(13)
    w = torch.randn(OUT_FEATURES, COLUMNS)
    mu = torch.randn(OUT_FEATURES)
    v = torch.rand(OUT_FEATURES) + 0.5
    correction = qbn_correction(w, torch.zeros_like(w), torch.rand(CHANNELS),
                                torch.rand(CHANNELS) + 1.0, mu, v, GROUP_SIZE)
    assert torch.allclose(correction.d_mean, torch.zeros(OUT_FEATURES), atol=1e-6)
    assert torch.allclose(correction.d_var, torch.zeros(OUT_FEATURES), atol=1e-6)
    assert torch.allclose(correction.running_mean, mu, atol=1e-6)
    assert torch.allclose(correction.running_var, v, atol=1e-6)
