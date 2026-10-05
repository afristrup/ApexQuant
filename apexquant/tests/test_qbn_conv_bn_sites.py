"""Topology adapter tests: the documented case split, and a forward-hook check."""

from __future__ import annotations

import collections

import pytest
import timm
import torch
from torch import nn
from torch.nn import functional as F

from apexquant.bn import (CASE_A, CASE_B, CASE_C, CASE_D,
                                 CORRECTABLE_CASES, bundled_activation,
                                 case_fractions, conv_bn_sites)
from apexquant.bn.activation_kind import activation_kind
from apexquant.bn.activation_kinds import ACT_IDENTITY, ACT_SILU, ACT_UNKNOWN

RESNET_CONVS = 53
RESNET_CASE_A = 32              # read ReLU(BN) directly
RESNET_BLOCK_ENTRY = 20         # bottleneck conv1 plus the four downsample convs
RESNET_CASE_C = 1               # reads the image, left dense
RESNET_STEM_FED = 2             # of the 20, the pair fed by the stem max pool
EFFICIENTNET_CONVS = 81
EFFICIENTNET_SILUS = 49
EFFICIENTNET_BUNDLED_SILUS = 33
EFFICIENTNET_SE_BLOCKS = 16
CONVNEXT_CONVS = 22
SPATIAL = 64
SEEDS = (0, 1, 2)
TOLERANCE = 1e-5
ACT_OF_KIND = {ACT_IDENTITY: lambda t: t, "relu": torch.relu, ACT_SILU: F.silu}


def _model(name):
    return timm.create_model(name, pretrained=False).eval()


def _cases(sites):
    return collections.Counter(site.case for site in sites)


def test_resnet50_reproduces_the_documented_32_20_1_split():
    sites = conv_bn_sites(_model("resnet50"))
    assert len(sites) == RESNET_CONVS
    cases = _cases(sites)
    entry = [s for s in sites if s.case in (CASE_B, CASE_D)]
    assert cases[CASE_A] == RESNET_CASE_A
    assert len(entry) == RESNET_BLOCK_ENTRY
    assert cases[CASE_C] == RESNET_CASE_C
    # The document's 20 is 18 genuine residual sums plus the two stem-fed
    # convolutions, which section 4.4 puts in case D because a max pool feeds them.
    assert cases[CASE_B] == RESNET_BLOCK_ENTRY - RESNET_STEM_FED
    assert cases[CASE_D] == RESNET_STEM_FED
    assert all(s.conv.endswith(("conv1", "downsample.0")) for s in entry)


def test_resnet50_case_d_pair_is_exactly_the_stem_max_pool():
    sites = conv_bn_sites(_model("resnet50"))
    stem_fed = [s for s in sites if s.case == CASE_D]
    assert [s.conv for s in stem_fed] == ["layer1.0.conv1", "layer1.0.downsample.0"]
    assert {s.source for s in stem_fed} == {"maxpool"}


def test_resnet50_case_a_producers_are_the_documented_conv_to_bn_pairs():
    sites = {s.conv: s for s in conv_bn_sites(_model("resnet50"))}
    assert sites["layer1.0.conv2"].producers == ("layer1.0.bn1",)
    assert sites["layer1.0.conv2"].bn == "layer1.0.bn2"
    assert sites["layer1.0.conv3"].producers == ("layer1.0.bn2",)
    assert sites["layer1.0.conv3"].bn == "layer1.0.bn3"
    assert sites["layer1.0.conv2"].activation == "relu"


def test_resnet50_case_b_sums_the_projection_shortcut_but_carries_the_identity_one():
    sites = {s.conv: s for s in conv_bn_sites(_model("resnet50"))}
    projected = sites["layer1.1.conv1"]
    assert projected.producers == ("layer1.0.bn3", "layer1.0.downsample.1")
    assert projected.identity_from == ()
    carried = sites["layer1.2.conv1"]
    assert carried.producers == ("layer1.1.bn3",)
    assert carried.identity_from == ("layer1.0",)


def test_convnext_tiny_has_zero_correctable_sites():
    sites = conv_bn_sites(_model("convnext_tiny"))
    assert len(sites) == CONVNEXT_CONVS
    assert all(s.bn is None for s in sites)
    assert [s for s in sites if s.case in CORRECTABLE_CASES] == []
    assert case_fractions(sites)[CASE_D] == pytest.approx(100.0)


def test_efficientnet_b0_reads_the_bundled_silu_and_never_assumes_relu():
    model = _model("efficientnet_b0")
    assert sum(isinstance(m, nn.SiLU) for m in model.modules()) == EFFICIENTNET_SILUS
    bundled = [bundled_activation(m) for m in model.modules()
               if isinstance(m, nn.BatchNorm2d)]
    assert bundled.count(ACT_SILU) == EFFICIENTNET_BUNDLED_SILUS
    sites = conv_bn_sites(model)
    correctable = [s for s in sites if s.case in CORRECTABLE_CASES]
    assert {s.activation for s in correctable} == {ACT_SILU, ACT_IDENTITY}


def test_efficientnet_b0_squeeze_excitation_fraction_is_reported():
    model = _model("efficientnet_b0")
    sites = conv_bn_sites(model)
    assert len(sites) == EFFICIENTNET_CONVS
    cases = _cases(sites)
    gated = [s for s in sites if s.case == CASE_D]
    # Two convolutions inside each gate, plus the projection reading the gated
    # tensor: three per squeeze-excitation block, and nothing else is case D.
    assert len(gated) == 3 * EFFICIENTNET_SE_BLOCKS
    assert cases[CASE_C] == 1
    assert cases[CASE_A] + cases[CASE_B] == EFFICIENTNET_CONVS - len(gated) - 1
    assert case_fractions(sites)[CASE_D] == pytest.approx(
        100.0 * len(gated) / EFFICIENTNET_CONVS)


@pytest.mark.parametrize("name", ["resnet50", "efficientnet_b0"])
@pytest.mark.parametrize("seed", SEEDS)
def test_forward_hooks_confirm_every_case_a_and_b_site(name, seed):
    """Randomised end-to-end check that the resolved producers are the real ones.

    Every BatchNorm gets random buffers and random affine parameters so that no
    two are interchangeable, then a random image is pushed through on CPU. For
    each case-A and case-B site the claimed producers, identity branches and
    activation must reconstruct the tensor the convolution actually received.
    """
    torch.manual_seed(seed)
    model = _randomized(_model(name))
    sites = conv_bn_sites(model)
    modules = dict(model.named_modules())
    outputs, inputs = _capture(model, [s.conv for s in sites], modules,
                               torch.randn(1, 3, SPATIAL, SPATIAL))
    checked = 0
    for site in sites:
        if site.case not in CORRECTABLE_CASES:
            continue
        got = _rebuilt_input(site, outputs, modules)
        assert torch.allclose(got, inputs[site.conv], atol=TOLERANCE), site
        checked += 1
    assert checked == sum(s.case in CORRECTABLE_CASES for s in sites)
    assert checked > 0


@pytest.mark.parametrize("name", ["resnet50", "efficientnet_b0"])
def test_bundled_activation_is_what_the_batchnorm_module_applies(name):
    """Falsify the activation claim against the module, not against its class name.

    A ``BatchNormAct2d`` folds the activation into its own output, so a site that
    named the wrong one would still reconstruct correctly downstream. This
    recomputes the plain normalization from the buffers and checks that the only
    thing standing between it and the module's real output is the activation the
    adapter claims. Naming ReLU where EfficientNet-B0 has SiLU fails here.
    """
    torch.manual_seed(SEEDS[0])
    model = _randomized(_model(name))
    modules = dict(model.named_modules())
    norms = [n for n, m in modules.items() if isinstance(m, nn.BatchNorm2d)]
    outputs, inputs = _capture(model, norms, modules,
                               torch.randn(1, 3, SPATIAL, SPATIAL))
    for norm in norms:
        bn = modules[norm]
        plain = F.batch_norm(inputs[norm], bn.running_mean, bn.running_var,
                             bn.weight, bn.bias, False, 0.0, bn.eps)
        expected = ACT_OF_KIND[bundled_activation(bn)](plain)
        assert torch.allclose(expected, outputs[norm], atol=TOLERANCE), norm


def test_case_c_site_receives_the_image():
    torch.manual_seed(SEEDS[0])
    model = _randomized(_model("resnet50"))
    sites = conv_bn_sites(model)
    image = torch.randn(1, 3, SPATIAL, SPATIAL)
    _, inputs = _capture(model, [s.conv for s in sites],
                         dict(model.named_modules()), image)
    stem = next(s for s in sites if s.case == CASE_C)
    assert torch.equal(inputs[stem.conv], image)


def test_unknown_family_with_batchnorm_fails_fast():
    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
    with pytest.raises(TypeError, match="no topology adapter"):
        conv_bn_sites(model)


def test_batchnorm_subclass_hiding_an_activation_fails_fast():
    class SneakyNorm(nn.BatchNorm2d):
        def __init__(self, num_features):
            super().__init__(num_features)
            self.act = nn.SiLU()

    with pytest.raises(TypeError, match="not a BatchNormAct2d"):
        bundled_activation(SneakyNorm(4))


def test_activation_subclass_is_not_mistaken_for_its_base():
    class ClippedRelu(nn.ReLU):
        pass

    assert activation_kind(ClippedRelu()) == ACT_UNKNOWN
    assert activation_kind(None) == ACT_IDENTITY


def _randomized(model):
    """Give every BatchNorm distinguishable buffers and affine parameters."""
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            module.running_mean.normal_()
            module.running_var.uniform_(0.5, 2.0)
            module.weight.normal_(1.0, 0.5)
            module.bias.normal_(0.0, 0.5)
    return model.eval()


def _capture(model, input_names, modules, image):
    """Run the model once, recording every module output and the named inputs."""
    outputs, inputs, handles = {}, {}, []

    def record_output(name):
        def hook(_module, _args, out):
            if isinstance(out, torch.Tensor):
                outputs[name] = out.detach().clone()
        return hook

    def record_input(name):
        def hook(_module, args):
            inputs[name] = args[0].detach().clone()
        return hook

    for name, module in modules.items():
        if name:
            handles.append(module.register_forward_hook(record_output(name)))
    for name in input_names:
        handles.append(
            modules[name].register_forward_pre_hook(record_input(name)))
    with torch.no_grad():
        model(image)
    for handle in handles:
        handle.remove()
    return outputs, inputs


def _rebuilt_input(site, outputs, modules):
    """Rebuild a site's input from its recorded producers and identity branches."""
    parts = [outputs[name] for name in site.producers + site.identity_from]
    total = parts[0]
    for part in parts[1:]:
        total = total + part
    if {bundled_activation(modules[n]) for n in site.producers} == {site.activation}:
        return total                      # the BatchNormAct2d already applied it
    return ACT_OF_KIND[site.activation](total)
