"""row_alpha and perp_energy: the per-row/per-group algebra of QBN section 4.5."""

import pytest
import torch

from apexquant.bn import perp_energy, row_alpha

OUT_FEATURES = 12
NUM_GROUPS = 5
GROUP_SIZE = 64
IN_FEATURES = NUM_GROUPS * GROUP_SIZE

# Monte Carlo sizes. The isotropy corollary has a relative standard deviation of
# order sqrt(2 / (gs + 2)) per draw, so a few 1e4 draws buy three digits.
ROTATION_DRAWS = 20000
SAMPLE_DRAWS = 400000


def _grouped(t):
    return t.view(t.shape[0], NUM_GROUPS, GROUP_SIZE)


def test_row_alpha_matches_an_explicit_per_row_loop():
    torch.manual_seed(0)
    w = torch.randn(OUT_FEATURES, IN_FEATURES)
    dw = 0.1 * torch.randn(OUT_FEATURES, IN_FEATURES)

    alpha = row_alpha(w, dw)

    expected = torch.tensor([
        float(torch.dot(w[c], dw[c]) / torch.dot(w[c], w[c])) for c in range(OUT_FEATURES)
    ])
    assert alpha.shape == (OUT_FEATURES,)
    assert torch.allclose(alpha, expected, rtol=1e-6, atol=1e-7)


def test_row_alpha_recovers_a_planted_pure_contraction():
    torch.manual_seed(1)
    w = torch.randn(OUT_FEATURES, IN_FEATURES)
    planted = torch.linspace(-0.2, 0.05, OUT_FEATURES)
    dw = planted[:, None] * w

    assert torch.allclose(row_alpha(w, dw), planted, rtol=1e-5, atol=1e-6)
    # A pure contraction has no orthogonal part left over.
    assert float(perp_energy(_grouped(w), _grouped(dw), planted).abs().max()) < 1e-6


def test_row_alpha_is_unchanged_by_the_structural_zero_padding():
    torch.manual_seed(2)
    w = torch.randn(OUT_FEATURES, IN_FEATURES)
    dw = 0.1 * torch.randn(OUT_FEATURES, IN_FEATURES)
    pad = torch.zeros(OUT_FEATURES, GROUP_SIZE)
    padded_dw = torch.cat([dw, 0.1 * torch.randn(OUT_FEATURES, GROUP_SIZE)], dim=1)

    alpha = row_alpha(w, dw)
    padded_alpha = row_alpha(torch.cat([w, pad], dim=1), padded_dw)

    assert torch.allclose(alpha, padded_alpha, rtol=1e-6, atol=1e-7)


def test_perp_is_orthogonal_to_the_row_and_sums_to_the_whole_row_form():
    torch.manual_seed(3)
    w = torch.randn(OUT_FEATURES, IN_FEATURES)
    dw = 0.1 * torch.randn(OUT_FEATURES, IN_FEATURES)
    alpha = row_alpha(w, dw)

    perp = dw - alpha[:, None] * w
    assert float((w * perp).sum(dim=1).abs().max()) < 1e-4

    energy = perp_energy(_grouped(w), _grouped(dw), alpha)
    assert energy.shape == (OUT_FEATURES, NUM_GROUPS)
    whole_row = perp.pow(2).sum(dim=1)
    assert torch.allclose(energy.sum(dim=1), whole_row, rtol=1e-5, atol=1e-6)
    # Pythagoras: ||dW_c||^2 = alpha_c^2 ||W_c||^2 + ||dWperp_c||^2.
    assert torch.allclose(
        dw.pow(2).sum(dim=1),
        alpha.pow(2) * w.pow(2).sum(dim=1) + whole_row,
        rtol=1e-5, atol=1e-6,
    )


def test_perp_energy_is_per_group_and_not_the_row_energy_spread_evenly():
    """The per-group decomposition carries information the row form throws away."""
    torch.manual_seed(4)
    w = torch.randn(OUT_FEATURES, IN_FEATURES)
    dw = 0.1 * torch.randn(OUT_FEATURES, IN_FEATURES)
    # Concentrate the error in one group, as a padded or badly conditioned group
    # would do in a real layer.
    dw.view(OUT_FEATURES, NUM_GROUPS, GROUP_SIZE)[:, 2] *= 8.0

    energy = perp_energy(_grouped(w), _grouped(dw), row_alpha(w, dw))
    share = energy[:, 2] / energy.sum(dim=1)
    assert float(share.min()) > 1.0 / NUM_GROUPS + 0.3


def test_isotropy_corollary_against_monte_carlo_rotations():
    """E[dwperp^T C dwperp] = ||dwperp||^2 tr(C)/d, per group, C diagonal."""
    torch.manual_seed(5)
    variance = torch.rand(GROUP_SIZE) * 3.0 + 0.05    # tau over this group's columns
    tau = float(variance.mean())

    direction = torch.randn(ROTATION_DRAWS, GROUP_SIZE)
    direction = direction / direction.norm(dim=1, keepdim=True)
    norm_squared = 0.7
    quadratic = (direction.pow(2) * variance).sum(dim=1) * norm_squared

    predicted = norm_squared * tau
    measured = float(quadratic.mean())
    assert abs(measured - predicted) / predicted < 0.02


def test_variance_decomposition_identity_against_monte_carlo_samples():
    """Var[Y'] = (1+a)^2 W C W + 2(1+a) W C dWperp + dWperp C dWperp, MC-checked."""
    torch.manual_seed(6)
    out_features, in_features = 6, 48
    w = torch.randn(out_features, in_features)
    dw = 0.15 * torch.randn(out_features, in_features)
    alpha = row_alpha(w, dw)
    perp = dw - alpha[:, None] * w

    mean = 0.4 * torch.randn(in_features)
    std = 0.3 + torch.rand(in_features)
    x = mean + std * torch.randn(SAMPLE_DRAWS, in_features)

    y = x @ w.T
    y_quantized = x @ (w + dw).T
    measured = y_quantized.var(dim=0, unbiased=False)

    cov = torch.diag(std.pow(2))
    v = (w @ cov * w).sum(dim=1)
    cross = 2.0 * (1.0 + alpha) * (w @ cov * perp).sum(dim=1)
    isotropic = (perp @ cov * perp).sum(dim=1)
    predicted = (1.0 + alpha).pow(2) * v + cross + isotropic

    assert torch.allclose(measured, predicted, rtol=0.02, atol=1e-4)
    # And the same identity holds exactly against the realized sample covariance,
    # which is what catches a sign error in the decomposition.
    centered = x - x.mean(dim=0)
    sample_cov = centered.T @ centered / SAMPLE_DRAWS
    exact = ((1.0 + alpha).pow(2) * (w @ sample_cov * w).sum(dim=1)
             + 2.0 * (1.0 + alpha) * (w @ sample_cov * perp).sum(dim=1)
             + (perp @ sample_cov * perp).sum(dim=1))
    assert torch.allclose(measured, exact, rtol=1e-4, atol=1e-5)


def test_row_alpha_fails_fast():
    w = torch.randn(OUT_FEATURES, IN_FEATURES)
    with pytest.raises(ValueError, match="2-D"):
        row_alpha(w.view(OUT_FEATURES, NUM_GROUPS, GROUP_SIZE), w.view(OUT_FEATURES, NUM_GROUPS, GROUP_SIZE))
    with pytest.raises(ValueError, match="dw must match"):
        row_alpha(w, w[:, :-1])
    dead = w.clone()
    dead[3] = 0.0
    with pytest.raises(ValueError, match="zero output row"):
        row_alpha(dead, w)


def test_perp_energy_fails_fast_on_a_bad_group_layout():
    torch.manual_seed(7)
    w = torch.randn(OUT_FEATURES, IN_FEATURES)
    alpha = row_alpha(w, w)

    with pytest.raises(ValueError, match="3-D"):
        perp_energy(w, w, alpha)
    with pytest.raises(ValueError, match="power of two"):
        odd = w[:, : 3 * 21].reshape(OUT_FEATURES, 3, 21)
        perp_energy(odd, odd, alpha)
    with pytest.raises(ValueError, match="alpha must be"):
        # A group spanning two output rows: the channel-last flattening of
        # section 8.3, which makes the decomposition ill posed per channel.
        spanning = w.reshape(OUT_FEATURES // 2, 2 * NUM_GROUPS, GROUP_SIZE)
        perp_energy(spanning, spanning, alpha)
