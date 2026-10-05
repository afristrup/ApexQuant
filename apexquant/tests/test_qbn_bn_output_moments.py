"""bn_output_moments: beta and |gamma| * sqrt(v / (v + eps)), including gamma < 0."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from apexquant.bn.bn_output_moments import bn_output_moments

EPS = 1e-5
CHANNELS = 8
SEED = 20260914


def _bn(gamma: torch.Tensor, beta: torch.Tensor, var: torch.Tensor) -> nn.BatchNorm2d:
    bn = nn.BatchNorm2d(gamma.numel(), eps=EPS)
    with torch.no_grad():
        bn.weight.copy_(gamma)
        bn.bias.copy_(beta)
        bn.running_var.copy_(var)
        bn.running_mean.copy_(torch.linspace(-1.0, 1.0, gamma.numel()))
    return bn.eval()


def test_moments_match_the_closed_form():
    torch.manual_seed(SEED)
    gamma = torch.randn(CHANNELS)
    beta = torch.randn(CHANNELS)
    var = torch.rand(CHANNELS) + 0.1

    m, sigma = bn_output_moments(_bn(gamma, beta, var))

    assert torch.allclose(m, beta)
    assert torch.allclose(sigma, gamma.abs() * torch.sqrt(var / (var + EPS)))


def test_negative_gamma_gives_a_positive_sigma():
    """0.39% of ResNet-50's BN channels have gamma < 0 and every published form
    of this formula uses gamma itself as a standard deviation."""
    gamma = torch.tensor([-2.5, -0.1, 0.1, 2.5])
    beta = torch.zeros(4)
    var = torch.ones(4)

    m, sigma = bn_output_moments(_bn(gamma, beta, var))

    assert bool((sigma > 0.0).all()), sigma
    assert torch.allclose(sigma[0], sigma[3])
    assert torch.allclose(sigma[1], sigma[2])
    assert torch.allclose(m, beta)


def test_zero_gamma_is_a_deterministic_channel():
    gamma = torch.tensor([0.0, 1.0])
    beta = torch.tensor([3.0, 3.0])
    _, sigma = bn_output_moments(_bn(gamma, beta, torch.ones(2)))

    assert sigma[0].item() == 0.0
    assert sigma[1].item() > 0.0


def test_moments_match_an_actual_forward_pass():
    """The formula describes what BatchNorm emits, so check it against emission."""
    torch.manual_seed(SEED)
    gamma = torch.tensor([-1.5, 0.5, 2.0, 0.0])
    beta = torch.tensor([0.25, -1.0, 0.0, 4.0])
    var = torch.tensor([0.5, 2.0, 1.0, 3.0])
    bn = _bn(gamma, beta, var)

    n = 400_000
    x = bn.running_mean.view(1, -1, 1, 1) + var.sqrt().view(1, -1, 1, 1) * torch.randn(
        n, gamma.numel(), 1, 1)
    with torch.no_grad():
        y = bn(x)

    m, sigma = bn_output_moments(bn)
    measured_m = y.mean(dim=(0, 2, 3))
    measured_sigma = y.std(dim=(0, 2, 3), unbiased=False)

    assert torch.allclose(m, measured_m, atol=2e-2)
    assert torch.allclose(sigma, measured_sigma, atol=2e-2)


def test_affine_false_is_the_identity_affine():
    bn = nn.BatchNorm2d(CHANNELS, eps=EPS, affine=False).eval()
    with torch.no_grad():
        bn.running_var.copy_(torch.rand(CHANNELS) + 0.1)

    m, sigma = bn_output_moments(bn)

    assert torch.allclose(m, torch.zeros(CHANNELS))
    assert torch.allclose(sigma, torch.sqrt(bn.running_var / (bn.running_var + EPS)))


def test_untracked_statistics_fail_fast():
    bn = nn.BatchNorm2d(CHANNELS, track_running_stats=False).eval()
    with pytest.raises(ValueError, match="running_var"):
        bn_output_moments(bn)


def test_negative_running_var_fails_fast():
    bn = nn.BatchNorm2d(CHANNELS).eval()
    with torch.no_grad():
        bn.running_var[0] = -1.0
    with pytest.raises(ValueError, match="negative"):
        bn_output_moments(bn)


def test_timm_batchnormact2d_is_accepted():
    timm = pytest.importorskip("timm")
    from timm.layers.norm_act import BatchNormAct2d

    model = timm.create_model("efficientnet_b0", pretrained=False).eval()
    layers = [mod for mod in model.modules() if isinstance(mod, BatchNormAct2d)]
    assert layers, "efficientnet_b0 should carry BatchNormAct2d"

    with torch.no_grad():
        layers[0].weight.copy_(-torch.abs(layers[0].weight) - 0.5)

    m, sigma = bn_output_moments(layers[0])

    assert m.shape == (layers[0].num_features,)
    assert bool((sigma > 0.0).all())
