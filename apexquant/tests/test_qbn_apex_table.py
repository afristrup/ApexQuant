"""Section 3.2: ApexQuant's error is anti-aligned with the row, alpha_c = -rho^{2P}."""

import pytest
import torch

from apexquant import ApexQuantConfig, dequantize_weight, turboquantize_weight
from apexquant.bn import perp_energy, row_alpha

# The design document's measured table: (group size, passes) -> (||dW_c||^2 /
# ||W_c||^2, alpha_c), averaged over random Gaussian rows at bits=2 with the
# "apex" density-matched codebook.
APEX_TABLE = {
    (256, 1): (0.11635, -0.11737),
    (256, 2): (0.01370, -0.01401),
    (256, 3): (0.00159, -0.00157),
    (512, 1): (0.11735, -0.11768),
    (512, 2): (0.01376, -0.01390),
    (512, 3): (0.00161, -0.00156),
}

BITS = 2
CODEBOOK = "apex"
TABLE_ROWS = 4096
WEIGHT_SEEDS = (0, 1, 2, 3)

# The table is a Monte Carlo average itself, so it is reproduced to a relative
# tolerance plus a floor that keeps the P=3 entries (a few 1e-3) honest.
TABLE_RTOL = 0.03
TABLE_ATOL = 1e-4

# Fraction of rows whose alpha_c must be negative, per pass count. The claim is
# that the contraction is systematic, not that it holds for literally every row:
# by P=3 the coherent part has shrunk to 1.6e-3 and the per-row spread straddles
# zero, exactly as the document's ResNet-50 table reports (realized alpha_c at
# P=3 has p90 = +0.0012, and 29.5% of channels get d_var_c > 0).
MIN_NEGATIVE_FRACTION = {1: 0.999, 2: 0.95, 3: 0.70}


def _measure(group_size, passes, seed):
    torch.manual_seed(seed)
    w = torch.randn(TABLE_ROWS, group_size)
    config = ApexQuantConfig(bits=BITS, codebook=CODEBOOK, group_size=group_size, passes=passes)
    dw = dequantize_weight(turboquantize_weight(w, config)) - w

    row_energy = w.pow(2).sum(dim=1)
    alpha = row_alpha(w, dw)
    grouped = (w.unsqueeze(1), dw.unsqueeze(1))     # (rows, 1 group, group_size)
    perp = perp_energy(*grouped, alpha).sum(dim=1)
    return dw.pow(2).sum(dim=1) / row_energy, alpha, perp / row_energy


@pytest.mark.parametrize("seed", WEIGHT_SEEDS)
@pytest.mark.parametrize("group_size,passes", sorted(APEX_TABLE))
def test_reproduces_the_section_3_2_table(group_size, passes, seed):
    expected_relative, expected_alpha = APEX_TABLE[(group_size, passes)]
    relative, alpha, _ = _measure(group_size, passes, seed)

    measured_relative = float(relative.mean())
    measured_alpha = float(alpha.mean())
    assert abs(measured_relative - expected_relative) <= TABLE_RTOL * expected_relative + TABLE_ATOL, (
        f"gs={group_size} P={passes} seed={seed}: "
        f"||dW_c||^2/||W_c||^2 = {measured_relative:.5f}, document says {expected_relative:.5f}"
    )
    assert abs(measured_alpha - expected_alpha) <= TABLE_RTOL * abs(expected_alpha) + TABLE_ATOL, (
        f"gs={group_size} P={passes} seed={seed}: "
        f"alpha_c = {measured_alpha:.5f}, document says {expected_alpha:.5f}"
    )


@pytest.mark.parametrize("group_size,passes", sorted(APEX_TABLE))
def test_alpha_is_systematically_negative(group_size, passes):
    """The central empirical claim: the output contracts, it does not inflate."""
    _, alpha, _ = _measure(group_size, passes, WEIGHT_SEEDS[0])

    assert float(alpha.mean()) < 0.0
    assert float(alpha.median()) < 0.0
    assert float((alpha < 0).float().mean()) >= MIN_NEGATIVE_FRACTION[passes]


@pytest.mark.parametrize("group_size,passes", sorted(APEX_TABLE))
def test_alpha_tracks_minus_the_relative_error(group_size, passes):
    """alpha_c = -rho^{2P} and ||dW_c||^2/||W_c||^2 = rho^{2P} are the same number."""
    relative, alpha, _ = _measure(group_size, passes, WEIGHT_SEEDS[0])

    rho_2p = float(relative.mean())
    assert abs(float(alpha.mean()) + rho_2p) <= TABLE_RTOL * rho_2p + TABLE_ATOL


@pytest.mark.parametrize("group_size,passes", sorted(APEX_TABLE))
def test_perp_energy_matches_its_closed_form(group_size, passes):
    """||dWperp_c||^2 = rho^{2P} (1 - rho^{2P}) ||W_c||^2, section 4.5."""
    relative, alpha, perp_relative = _measure(group_size, passes, WEIGHT_SEEDS[0])

    rho_2p = float(relative.mean())
    closed_form = rho_2p * (1.0 - rho_2p)
    assert abs(float(perp_relative.mean()) - closed_form) <= TABLE_RTOL * closed_form + TABLE_ATOL
    # Pythagoras again, this time against the realized quantizer.
    assert torch.allclose(perp_relative, relative - alpha.pow(2), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("group_size,passes", sorted(APEX_TABLE))
def test_the_reconstruction_is_a_contraction(group_size, passes):
    """||W_hat_c||^2 = (1 - rho^{2P}) ||W_c||^2, so folding cannot repair it."""
    torch.manual_seed(WEIGHT_SEEDS[0])
    w = torch.randn(TABLE_ROWS, group_size)
    config = ApexQuantConfig(bits=BITS, codebook=CODEBOOK, group_size=group_size, passes=passes)
    w_hat = dequantize_weight(turboquantize_weight(w, config))

    ratio = float((w_hat.pow(2).sum(dim=1) / w.pow(2).sum(dim=1)).mean())
    expected = 1.0 - APEX_TABLE[(group_size, passes)][0]
    assert ratio < 1.0
    assert abs(ratio - expected) <= TABLE_RTOL * APEX_TABLE[(group_size, passes)][0] + TABLE_ATOL
