"""End-to-end 2-bit (and 8-bit) quantization through the whole pipeline."""

import math

import pytest
import torch

from apexquant import (
    ApexQuantConfig,
    dequantize_weight,
    fused_dequant_matmul,
    turboquantize_weight,
)
from apexquant.codebook import apex_beta_levels, lloyd_max_levels
from apexquant.quantize import pack_bits, unpack_bits
from apexquant.quantize.dequantize_from_indices import dequantize_from_indices
from apexquant.quantize.quantize_to_indices import quantize_to_indices


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_pack_unpack_roundtrip_any_width(bits):
    idx = torch.randint(0, 2 ** bits, (3, 5, 128), dtype=torch.uint8)
    assert torch.equal(unpack_bits(pack_bits(idx, bits), bits), idx)


def test_pack_bits_matches_pack_int4_at_4bit():
    from apexquant.quantize import pack_int4
    idx = torch.randint(0, 16, (4, 64), dtype=torch.uint8)
    assert torch.equal(pack_bits(idx, 4), pack_int4(idx))


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_weight_roundtrip_and_compression(bits):
    torch.manual_seed(0)
    w = torch.randn(256, 512)
    qw = turboquantize_weight(w, ApexQuantConfig(bits=bits, passes=1))
    recon = dequantize_weight(qw)
    assert recon.shape == w.shape
    # Compression tracks the bit width (~16 / bits vs fp32, minus scale overhead).
    ratio = w.numel() * w.element_size() / qw.nbytes()
    assert ratio > (32 / bits) * 0.6


def test_2bit_fused_matches_dequantized_matmul():
    torch.manual_seed(0)
    w = torch.randn(128, 256)
    x = torch.randn(8, 256)
    qw = turboquantize_weight(w, ApexQuantConfig(bits=2, passes=2))
    fused = fused_dequant_matmul(x, qw, force_backend="torch")
    reference = x @ dequantize_weight(qw).T
    assert torch.allclose(fused, reference, atol=1e-3, rtol=1e-3)


def test_2bit_apex_codebook_beats_gaussian_on_matched_density():
    # At 2 bits (4 levels) the density-matched codebook is the whole game.
    d = 256
    g = torch.Generator().manual_seed(0)
    v = torch.randn(200_000, d, generator=g)
    x = (v / v.norm(dim=1, keepdim=True))[:, 0]  # ~ f_d

    def mse(levels):
        return (dequantize_from_indices(quantize_to_indices(x, levels), levels) - x).pow(2).mean().item()

    apex = apex_beta_levels(2, fan_in=d)
    gauss = lloyd_max_levels(2) / math.sqrt(d)  # gaussian rescaled to f_d variance
    assert mse(apex) <= mse(gauss) + 1e-9


def test_2bit_apex_pipeline_end_to_end():
    torch.manual_seed(0)
    w = torch.randn(128, 512)
    qw = turboquantize_weight(w, ApexQuantConfig(bits=2, passes=2, codebook="apex"))
    x = torch.randn(4, 512)
    fused = fused_dequant_matmul(x, qw, force_backend="torch")
    assert torch.allclose(fused, x @ dequantize_weight(qw).T, atol=1e-3, rtol=1e-3)
