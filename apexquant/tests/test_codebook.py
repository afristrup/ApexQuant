import torch

from apexquant.codebook import lloyd_max_levels
from apexquant.quantize import dequantize_from_indices, quantize_to_indices


def test_levels_shape_sorted_symmetric():
    levels = lloyd_max_levels(4)
    assert levels.shape == (16,)
    assert torch.all(levels[1:] > levels[:-1])  # strictly increasing
    # Gaussian is symmetric, so levels should be (anti)symmetric about 0.
    assert torch.allclose(levels, -levels.flip(0), atol=1e-3)


def test_quantization_mse_near_lloyd_max_optimum():
    torch.manual_seed(0)
    x = torch.randn(200_000)
    levels = lloyd_max_levels(4)
    recon = dequantize_from_indices(quantize_to_indices(x, levels), levels)
    mse = (recon - x).pow(2).mean().item()
    # Known 4-bit Lloyd-Max distortion for a unit Gaussian is ~0.0095.
    assert mse < 0.012
