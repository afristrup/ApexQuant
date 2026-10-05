"""The input-moment oracle of QBN section 4.4, against Monte Carlo and real topologies."""

from __future__ import annotations

import pytest
import timm
import torch
from torch import nn

from apexquant.bn.activation_kinds import (ACT_IDENTITY, ACT_IMAGE,
                                                  ACT_RELU, ACT_SILU)
from apexquant.bn.bn_output_moments import bn_output_moments
from apexquant.bn.case_letters import CASE_A, CASE_B, CASE_C, CASE_D
from apexquant.bn.conv_bn_site import ConvBnSite
from apexquant.bn.conv_bn_sites import conv_bn_sites
from apexquant.bn.input_moments import input_moments
from apexquant.bn.pre_activation_moments import pre_activation_moments

SAMPLES = 2_000_000
MC_TOLERANCE = 4e-3
CHANNELS = 5


class Branches(nn.Module):
    """Two BatchNorms and the convolution that reads their sum."""

    def __init__(self) -> None:
        super().__init__()
        self.bn_a = nn.BatchNorm2d(CHANNELS)
        self.bn_b = nn.BatchNorm2d(CHANNELS)
        self.conv = nn.Conv2d(CHANNELS, 3, 1, bias=False)
        with torch.no_grad():
            self.bn_a.weight.copy_(torch.tensor([1.0, -0.5, 2.0, 0.3, 0.0]))
            self.bn_a.bias.copy_(torch.tensor([0.2, -1.0, 0.5, 0.0, 1.0]))
            self.bn_a.running_var.copy_(torch.tensor([1.0, 4.0, 0.25, 2.0, 1.0]))
            self.bn_b.weight.copy_(torch.tensor([0.7, 1.1, -1.3, 0.9, 2.0]))
            self.bn_b.bias.copy_(torch.tensor([-0.3, 0.4, 0.0, 1.2, -0.6]))
            self.bn_b.running_var.copy_(torch.tensor([0.5, 1.0, 3.0, 1.0, 0.2]))
        self.eval()


def _sample(bn: nn.BatchNorm2d, seed: int) -> torch.Tensor:
    """Draw from the Gaussian ``bn_output_moments`` claims this BatchNorm emits."""
    m, sigma = bn_output_moments(bn)
    generator = torch.Generator().manual_seed(seed)
    return m + sigma * torch.randn(SAMPLES, CHANNELS, generator=generator)


def _assert_moments(e: torch.Tensor, s: torch.Tensor, x: torch.Tensor) -> None:
    scale = torch.maximum(torch.ones_like(s), (x * x).mean(dim=0))
    assert bool(((e - x.mean(dim=0)).abs() < MC_TOLERANCE * scale).all())
    assert bool(((s - (x * x).mean(dim=0)).abs() < MC_TOLERANCE * scale).all())


def test_case_a_relu_matches_monte_carlo():
    """Case A: read one BatchNorm, push it through the activation, and sample it."""
    model = Branches()
    site = ConvBnSite("conv", None, CASE_A, ("bn_a",), (), ACT_RELU, "node")
    e, s = input_moments(model, [site])["conv"]
    _assert_moments(e, s, torch.relu(_sample(model.bn_a, seed=11)))


def test_case_a_silu_matches_monte_carlo():
    """The quadrature path is the one EfficientNet-B0 takes for all 33 of its sites."""
    model = Branches()
    site = ConvBnSite("conv", None, CASE_A, ("bn_a",), (), ACT_SILU, "node")
    e, s = input_moments(model, [site])["conv"]
    _assert_moments(e, s, torch.nn.functional.silu(_sample(model.bn_a, seed=12)))


def test_case_a_identity_matches_monte_carlo():
    """EfficientNet-B0's project convolutions read a BatchNorm with no activation."""
    model = Branches()
    site = ConvBnSite("conv", None, CASE_A, ("bn_a",), (), ACT_IDENTITY, "node")
    e, s = input_moments(model, [site])["conv"]
    _assert_moments(e, s, _sample(model.bn_a, seed=13))


def test_case_b_sums_variances_then_activates():
    """Case B adds branch means and branch variances, then applies the activation."""
    model = Branches()
    site = ConvBnSite("conv", None, CASE_B, ("bn_a", "bn_b"), (), ACT_RELU, "node")
    e, s = input_moments(model, [site])["conv"]
    total = _sample(model.bn_a, seed=21) + _sample(model.bn_b, seed=22)
    _assert_moments(e, s, torch.relu(total))


def test_case_b_activating_each_branch_first_would_be_a_different_answer():
    """Order matters: sum then activate, never activate then sum."""
    model = Branches()
    site = ConvBnSite("conv", None, CASE_B, ("bn_a", "bn_b"), (), ACT_RELU, "node")
    e, _ = input_moments(model, [site])["conv"]
    wrong = torch.relu(_sample(model.bn_a, seed=21)) + torch.relu(_sample(model.bn_b, seed=22))
    assert not bool(((e - wrong.mean(dim=0)).abs() < MC_TOLERANCE).all())


def test_case_b_identity_branch_reuses_a_recorded_node():
    """An identity branch contributes ``m += e_n`` and ``sigma^2 += s_n - e_n^2``.

    The Monte Carlo draws the identity branch from the Gaussian carrying the
    recorded moments, not from the rectified distribution they came from,
    because that substitution *is* the model: section 4.4 keeps one
    ``(m, sigma)`` pair per branch and has nowhere to put a third moment. What is
    checked here is that the recorded pair is read, converted from raw second
    moment to variance, and added -- not that the Gaussian assumption holds.
    ``test_identity_branch_gaussian_assumption_is_an_approximation`` measures the
    error that assumption costs.
    """
    model = Branches()
    first = ConvBnSite("conv", None, CASE_A, ("bn_a",), (), ACT_RELU, "node0")
    second = ConvBnSite("conv", None, CASE_B, ("bn_b",), ("node0",), ACT_RELU, "node1")
    moments = input_moments(model, [first, second])
    e, s = moments["conv"]
    node_e, node_s = input_moments(model, [first])["conv"]
    generator = torch.Generator().manual_seed(31)
    identity = node_e + (node_s - node_e * node_e).clamp_min(0.0).sqrt() \
        * torch.randn(SAMPLES, CHANNELS, generator=generator)
    total = identity + _sample(model.bn_b, seed=32)
    _assert_moments(e, s, torch.relu(total))


def test_identity_branch_gaussian_assumption_is_an_approximation():
    """Section 8.2's testable weakness, measured rather than assumed away.

    A ResNet shortcut is a post-ReLU tensor and is not Gaussian, so the case-B
    sum is modelled with the right first two moments and the wrong shape. The
    error that costs is small but real, and it is the reason test 7.2 splits
    prediction quality by case.
    """
    model = Branches()
    first = ConvBnSite("conv", None, CASE_A, ("bn_a",), (), ACT_RELU, "node0")
    second = ConvBnSite("conv", None, CASE_B, ("bn_b",), ("node0",), ACT_RELU, "node1")
    e, _ = input_moments(model, [first, second])["conv"]
    truth = torch.relu(torch.relu(_sample(model.bn_a, seed=41)) + _sample(model.bn_b, seed=42))
    error = (e - truth.mean(dim=0)).abs()
    assert float(error.max()) > MC_TOLERANCE
    assert float(error.max()) < 0.1


def test_identity_branch_without_a_recorded_node_fails_fast():
    """A forward-order violation must raise, not silently drop a branch."""
    model = Branches()
    site = ConvBnSite("conv", None, CASE_B, ("bn_b",), ("never-seen",), ACT_RELU, "n")
    with pytest.raises(ValueError, match="has not been recorded"):
        input_moments(model, [site])


def test_negative_gamma_channels_use_the_absolute_value():
    """Channel 1 of ``bn_a`` has gamma = -0.5; a signed sigma would break ReLU's moments."""
    model = Branches()
    site = ConvBnSite("conv", None, CASE_A, ("bn_a",), (), ACT_RELU, "node")
    e, s = input_moments(model, [site])["conv"]
    assert bool((e >= 0.0).all())
    assert bool((s >= e * e - 1e-6).all())


def test_zero_gamma_channel_is_deterministic():
    """Channel 4 of ``bn_a`` has gamma = 0, so ReLU of it is exactly ``relu(beta)``."""
    model = Branches()
    site = ConvBnSite("conv", None, CASE_A, ("bn_a",), (), ACT_RELU, "node")
    e, s = input_moments(model, [site])["conv"]
    assert float(e[4]) == pytest.approx(1.0)
    assert float(s[4]) == pytest.approx(1.0)


@pytest.mark.parametrize("case,activation", [(CASE_C, ACT_IMAGE), (CASE_D, ACT_RELU)])
def test_uncorrectable_cases_get_no_entry(case, activation):
    """Case C is asserted elsewhere and case D is counted; neither gets moments here."""
    model = Branches()
    site = ConvBnSite("conv", None, case, (), (), activation, "node")
    assert input_moments(model, [site]) == {}
    with pytest.raises(ValueError):
        pre_activation_moments(model, site, {})


def test_channel_count_mismatch_fails_fast():
    """A producer of the wrong width means the adapter and the model disagree."""
    model = Branches()
    model.add_module("bn_wide", nn.BatchNorm2d(CHANNELS + 1))
    site = ConvBnSite("conv", None, CASE_A, ("bn_wide",), (), ACT_RELU, "node")
    with pytest.raises(ValueError, match="input channels"):
        input_moments(model, [site])


def test_case_a_with_two_producers_fails_fast():
    """The case letter and the producer count must agree or the walk is wrong."""
    model = Branches()
    site = ConvBnSite("conv", None, CASE_A, ("bn_a", "bn_b"), (), ACT_RELU, "n")
    with pytest.raises(ValueError, match="exactly"):
        input_moments(model, [site])


def test_resnet50_every_correctable_site_gets_well_formed_moments():
    """Fifty of ResNet-50's 53 convolutions are correctable and all must resolve."""
    model = timm.create_model("resnet50", pretrained=False).eval()
    sites = conv_bn_sites(model)
    moments = input_moments(model, sites)
    correctable = [s for s in sites if s.case in (CASE_A, CASE_B)]
    assert len(correctable) == 50
    assert set(moments) == {s.conv for s in correctable}
    for site in correctable:
        e, s = moments[site.conv]
        conv = model.get_submodule(site.conv)
        assert e.shape == (conv.in_channels,)
        assert bool((e >= 0.0).all()), site.conv
        assert bool((s - e * e >= -1e-6).all()), site.conv


def test_efficientnet_b0_resolves_through_the_quadrature_path():
    """EfficientNet-B0 has SiLU and Sigmoid and no ReLU, so every site is quadrature."""
    model = timm.create_model("efficientnet_b0", pretrained=False).eval()
    sites = conv_bn_sites(model)
    moments = input_moments(model, sites)
    correctable = [s for s in sites if s.case in (CASE_A, CASE_B)]
    assert {s.activation for s in correctable} <= {ACT_SILU, ACT_IDENTITY}
    assert set(moments) == {s.conv for s in correctable}
    for site in correctable:
        e, s = moments[site.conv]
        assert e.shape == (model.get_submodule(site.conv).in_channels,)
        assert bool(torch.isfinite(e).all()) and bool(torch.isfinite(s).all())


def test_convnext_tiny_has_no_correctable_site():
    """The negative control has no BatchNorm at all, so the oracle returns nothing."""
    model = timm.create_model("convnext_tiny", pretrained=False).eval()
    assert input_moments(model, conv_bn_sites(model)) == {}
