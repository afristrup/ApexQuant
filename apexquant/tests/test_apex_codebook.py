"""Tests for the ApexQuant density-matched codebook and its integration."""

import math

import torch

from apexquant import ApexQuantConfig, dequantize_weight, turboquantize_weight
from apexquant.codebook import apex_beta_levels, lloyd_max_levels
from apexquant.quantize import dequantize_from_indices, quantize_to_indices


def _sample_fd(fan_in, n, seed=0):
    """Draw coordinates ~ f_d: first coord of a uniform point on S^{d-1}."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, fan_in, generator=g)
    return (v / v.norm(dim=1, keepdim=True))[:, 0]


def test_levels_sorted_symmetric_in_unit_interval():
    levels = apex_beta_levels(4, fan_in=128)
    assert levels.shape == (16,)
    assert torch.all(levels[1:] > levels[:-1])
    assert levels.min() >= -1 and levels.max() <= 1
    assert torch.allclose(levels, -levels.flip(0), atol=1e-3)


def test_d3_degenerates_to_uniform_grid():
    # f_3 is uniform on [-1,1]; Lloyd-Max cell centroids are then evenly spaced.
    levels = apex_beta_levels(3, fan_in=3)  # 8 levels
    gaps = (levels[1:] - levels[:-1])
    assert torch.allclose(gaps, gaps.mean(), atol=2e-2)  # near-uniform spacing


def test_large_fan_in_approaches_gaussian_lloyd_max():
    # f_d scaled by sqrt(d) -> N(0,1), so apex levels * sqrt(d) -> gaussian levels.
    d = 8192
    apex = apex_beta_levels(3, fan_in=d) * math.sqrt(d)
    gauss = lloyd_max_levels(3)
    assert (apex - gauss).abs().max() < 0.1


def test_matched_codebook_beats_uniform_and_gaussian_at_2bit():
    d = 256
    x = _sample_fd(d, 200_000)

    def mse(levels):
        recon = dequantize_from_indices(quantize_to_indices(x, levels), levels)
        return (recon - x).pow(2).mean().item()

    apex = apex_beta_levels(2, fan_in=d)
    uniform = torch.linspace(-1, 1, 4)
    gauss = lloyd_max_levels(2) / math.sqrt(d)  # gaussian levels rescaled to f_d
    m_apex, m_uniform, m_gauss = mse(apex), mse(uniform), mse(gauss)
    assert m_apex < m_uniform
    assert m_apex <= m_gauss + 1e-9


def test_apex_pipeline_roundtrip_matches_or_beats_gaussian():
    torch.manual_seed(0)
    w = torch.randn(256, 512)
    err_g = (dequantize_weight(turboquantize_weight(
        w, ApexQuantConfig(passes=1, codebook="gaussian"))) - w).norm() / w.norm()
    err_a = (dequantize_weight(turboquantize_weight(
        w, ApexQuantConfig(passes=1, codebook="apex"))) - w).norm() / w.norm()
    # Apex is matched to the exact post-rotation density; should be no worse.
    assert err_a <= err_g + 5e-3
