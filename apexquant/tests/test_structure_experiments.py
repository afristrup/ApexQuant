"""Tests for the HQQ/SINQ-derived stage and preprocessing variants.

These cover the four things Section 6 of the paper proposes: measuring the error
tail, fitting the stage under an l_p objective, reshaping the trellis generator
marginal, and Sinkhorn-normalizing before the rotation. Each assertion here is a
property the construction must have, not a result we hope for; the results
themselves live in ``experiments/structure_experiments.py``.
"""

import math

import torch

from apexquant import (
    ApexQuantConfig,
    dequantize_weight,
    error_tail_statistics,
    generalized_gaussian_levels,
    lp_levels,
    sinkhorn_scales,
    turboquantize_weight,
)
from apexquant.codebook import apex_beta_levels, lloyd_max_levels, trellis_quantize


def _sample_fd(fan_in, n, seed=0):
    """Draw coordinates ~ f_d: first coord of a uniform point on S^{d-1}."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, fan_in, generator=g)
    return (v / v.norm(dim=1, keepdim=True))[:, 0]


# --------------------------------------------------------------------------
# Tail statistics: the instrument, calibrated against densities we know.
# --------------------------------------------------------------------------

def test_tail_statistics_calibrate_on_known_densities():
    g = torch.Generator().manual_seed(0)
    gauss = error_tail_statistics(torch.randn(200_000, generator=g))
    assert abs(gauss["excess_kurtosis"]) < 0.15
    assert abs(gauss["gg_shape"] - 2.0) < 0.15

    # Laplace has excess kurtosis exactly 3 and generalized-Gaussian shape 1.
    lap = torch.distributions.Laplace(0.0, 1.0).sample((200_000,))
    lap_stats = error_tail_statistics(lap)
    assert abs(lap_stats["excess_kurtosis"] - 3.0) < 0.6
    assert abs(lap_stats["gg_shape"] - 1.0) < 0.2
    # A heavier tail must concentrate more of the squared error in the top slice.
    assert lap_stats["top_share"] > gauss["top_share"]


def test_tail_statistics_reject_degenerate_input():
    zero = error_tail_statistics(torch.zeros(64))
    assert zero["top_share"] == 0.0
    try:
        error_tail_statistics(torch.tensor([1.0]))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for a single element")


# --------------------------------------------------------------------------
# l_p stage codebook (HQQ's objective, our recursion).
# --------------------------------------------------------------------------

def test_lp_levels_reduce_to_the_mse_codebook_at_p2():
    # p = 2 is ordinary Lloyd-Max, so this must reproduce apex exactly enough
    # that the l_p path is a strict generalization rather than a second codebook.
    assert torch.allclose(lp_levels(2, 512, p=2.0), apex_beta_levels(2, 512), atol=1e-4)
    assert torch.allclose(lp_levels(3, 128, p=2.0), apex_beta_levels(3, 128), atol=1e-4)


def test_lp_levels_are_sorted_symmetric_and_bounded():
    levels = lp_levels(3, 256, p=0.7)
    assert levels.shape == (8,)
    assert torch.all(levels[1:] > levels[:-1])
    assert levels.min() >= -1.0 and levels.max() <= 1.0
    assert torch.allclose(levels, -levels.flip(0), atol=1e-3)


def test_lp_stage_trades_squared_error_for_its_own_objective():
    # The whole point of the swap: worse in L2 by construction (so rho rises),
    # better on the l_p objective it was fitted to. If either half fails the
    # comparison in the paper is not measuring what it claims to.
    d, p = 512, 0.7
    u = _sample_fd(d, 200_000, seed=1)
    mse_cb, lp_cb = apex_beta_levels(2, d), lp_levels(2, d, p=p)

    def err(levels):
        q = levels[(u[:, None] - levels[None, :]).abs().argmin(1)]
        return u - q

    e_mse, e_lp = err(mse_cb), err(lp_cb)
    assert e_lp.pow(2).mean() > e_mse.pow(2).mean()                 # rho goes up
    assert e_lp.abs().pow(p).mean() < e_mse.abs().pow(p).mean()     # l_p goes down


def test_lp_codebook_runs_end_to_end_through_the_quantizer():
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    qw = turboquantize_weight(w, ApexQuantConfig(bits=2, group_size=128, passes=2,
                                                 codebook="lp", lp_norm=0.7))
    assert qw.levels.numel() == 4
    rel = (dequantize_weight(qw) - w).norm() / w.norm()
    assert rel < 0.5  # a working two-pass 2-bit artifact, not a broken one


# --------------------------------------------------------------------------
# 1MAD-p: keep the trellis, reshape the generator marginal.
# --------------------------------------------------------------------------

def test_generalized_gaussian_reduces_to_gaussian_at_shape_2():
    assert torch.allclose(generalized_gaussian_levels(3, 2.0), lloyd_max_levels(3), atol=5e-3)


def test_generalized_gaussian_levels_have_unit_variance_and_heavier_tails():
    for shape in (2.0, 1.0, 0.7):
        levels = generalized_gaussian_levels(6, shape)
        assert torch.all(levels[1:] > levels[:-1])
    # Lower shape means more codeword mass pushed away from the origin.
    span = [generalized_gaussian_levels(6, s).abs().max().item() for s in (2.0, 1.0, 0.7)]
    assert span[0] < span[1] < span[2]


def test_shaped_trellis_keeps_the_trellis_and_only_moves_rho():
    torch.manual_seed(0)
    x = torch.randn(2048)
    base = trellis_quantize(x, 2)
    shaped = trellis_quantize(x, 2, fine=generalized_gaussian_levels(3, 0.7))
    assert base.shape == shaped.shape == x.shape
    # Heavier-tailed generator is worse in squared error, by construction.
    assert (x - shaped).pow(2).mean() > (x - base).pow(2).mean()
    # Passing the Gaussian marginal explicitly must reproduce the default path.
    explicit = trellis_quantize(x, 2, fine=lloyd_max_levels(3))
    assert torch.allclose(explicit, base)


def test_shaped_trellis_rejects_a_wrong_sized_generator():
    try:
        trellis_quantize(torch.randn(16), 2, fine=torch.linspace(-1, 1, 5))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for a mis-sized fine codebook")


# --------------------------------------------------------------------------
# Sinkhorn before the rotation (SINQ).
# --------------------------------------------------------------------------

def test_sinkhorn_equalizes_row_and_column_variance():
    torch.manual_seed(0)
    w = torch.randn(128, 256) * torch.exp(torch.randn(256) * 1.2)
    row, col = sinkhorn_scales(w, iters=10)
    a = w / (row[:, None] * col)
    # Both marginals land at unit std, which is the fixed point being solved for.
    assert (a.pow(2).mean(dim=1).sqrt() - 1.0).abs().max() < 1e-3
    assert (a.pow(2).mean(dim=0).sqrt() - 1.0).abs().max() < 1e-3
    assert torch.isfinite(row).all() and torch.isfinite(col).all()


def test_sinkhorn_gauge_is_pinned_to_the_columns():
    # row *= t, col /= t leaves the factorization unchanged, so the split is only
    # well defined once a gauge is fixed; ours puts geometric mean 1 on col.
    torch.manual_seed(0)
    w = torch.randn(64, 128) * 7.5
    _, col = sinkhorn_scales(w, iters=10)
    assert abs(col.log().mean().exp().item() - 1.0) < 1e-4


def test_sinkhorn_round_trip_reconstructs_the_original_scale():
    torch.manual_seed(0)
    w = torch.randn(64, 256) * torch.exp(torch.randn(256))
    qw = turboquantize_weight(w, ApexQuantConfig(bits=8, group_size=128, passes=2,
                                                 codebook="apex", sinkhorn_iters=10))
    assert qw.row_scale is not None and qw.col_scale is not None
    # At 8 bits x 2 passes the artifact is near-exact, so any error in undoing the
    # preprocessing would show up here as a gross scale mismatch.
    assert (dequantize_weight(qw) - w).norm() / w.norm() < 0.02


def test_sinkhorn_is_a_no_op_on_an_already_balanced_matrix():
    torch.manual_seed(0)
    w = torch.randn(128, 256)
    cfg = dict(bits=2, group_size=128, passes=2, codebook="apex")
    plain = (dequantize_weight(turboquantize_weight(w, ApexQuantConfig(**cfg))) - w).norm()
    sink = (dequantize_weight(
        turboquantize_weight(w, ApexQuantConfig(**cfg, sinkhorn_iters=10))) - w).norm()
    assert abs(plain / sink - 1.0) < 0.02


def test_sinkhorn_equalizes_per_column_relative_error():
    # This is the mechanism, and it is what makes the scheme activation-aware
    # without data: per-group normalization lets a few large columns dominate
    # their group, so the small columns in that group carry a huge *relative*
    # error. Sinkhorn removes the imbalance before the rotation ever sees it.
    torch.manual_seed(0)
    w = torch.randn(128, 512) * torch.exp(torch.randn(512) * 1.2)
    cfg = dict(bits=2, group_size=128, passes=2, codebook="apex")

    def spread(**kw):
        qw = turboquantize_weight(w, ApexQuantConfig(**cfg, **kw))
        rel = (dequantize_weight(qw) - w).norm(dim=0) / w.norm(dim=0).clamp_min(1e-12)
        return (rel.std() / rel.mean()).item()

    assert spread(sinkhorn_iters=10) < 0.2 * spread()


def test_sinkhorn_scales_survive_the_fused_linear_path():
    # The two diagonals never enter the packed kernel; they scale the activation
    # and the output instead. That has to agree with the dense reconstruction.
    from torch import nn

    from apexquant import ApexQuantLinear

    torch.manual_seed(0)
    linear = nn.Linear(256, 64, bias=True)
    with torch.no_grad():
        linear.weight.mul_(torch.exp(torch.randn(256)))
    cfg = ApexQuantConfig(bits=4, group_size=128, passes=2, codebook="apex",
                           sinkhorn_iters=10)
    fused = ApexQuantLinear.from_linear(linear, cfg)

    x = torch.randn(8, 256)
    qw = turboquantize_weight(linear.weight.detach(), cfg)
    reference = x @ dequantize_weight(qw).T + linear.bias
    assert torch.allclose(fused(x), reference, atol=1e-3, rtol=1e-3)


def test_sinkhorn_is_counted_in_the_stored_payload():
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    cfg = dict(bits=2, group_size=128, passes=1, codebook="apex")
    plain = turboquantize_weight(w, ApexQuantConfig(**cfg))
    sink = turboquantize_weight(w, ApexQuantConfig(**cfg, sinkhorn_iters=10))
    extra = sink.nbytes() - plain.nbytes()
    assert extra == 4 * (64 + 256)  # two float32 vectors, nothing hidden
