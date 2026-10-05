"""The QBN driver of sections 4.5 to 4.7, end to end on CPU."""

from __future__ import annotations

import importlib.util
import pathlib

import pytest
import timm
import torch
from torch import nn

from apexquant.bn.activation_kinds import ACT_RELU
from apexquant.bn.apply_qbn import apply_qbn
from apexquant.bn.case_letters import CASE_A, CASE_B, CASE_C, CASE_D
from apexquant.bn.conv_bn_site import ConvBnSite
from apexquant.bn.harness_group_size import (MAX_GROUP, MIN_FANIN_GROUP,
                                                    harness_group_size)
from apexquant.bn.input_moments import input_moments
from apexquant.bn.layer_residual import LayerResidual
from apexquant.bn.qbn_correction import qbn_correction
from apexquant.bn.qbn_guards import (SKIP_NO_RESIDUAL, SKIP_NONE,
                                            SKIP_UNCORRECTABLE_CASE)
from apexquant.bn.qbn_modes import MODE_MEAN, MODE_MEAN_VAR, MODE_VAR
from apexquant.bn.quantize_with_residuals import quantize_with_residuals
from apexquant.bn.row_alpha import row_alpha

PASSES = 2
IMAGES = 256
BATCH = 64
STEM_IN = 32
MID = 16
OUT = 64
SIDE = 8
FAN_IN = MID * 3 * 3

#: One float32 rounding of a subtraction, with room for the whole tensor's range.
FLOAT32_ROUNDTRIP = 1e-6

HARNESS = pathlib.Path(__file__).resolve().parents[1] / "experiments" \
    / "progressive_vision_imagenet.py"


def _load_harness():
    """Import the paper's quantization script without running it."""
    spec = importlib.util.spec_from_file_location("progressive_vision_imagenet", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stack(nn.Module):
    """conv-bn-relu-conv-bn, with the first stage's output exactly controllable.

    ``conv1`` has orthonormal rows and a 1x1 kernel, so with white input its
    output channels are independent and unit variance at every pixel. ``bn1``'s
    stored statistics are then exactly right, its output is exactly
    ``N(beta, gamma^2 v/(v+eps))``, and the rectified tensor that ``conv2`` reads
    has independent coordinates. ``conv2`` uses no padding, so no receptive field
    ever contains a structural zero and spatial stationarity is exact.
    """

    def __init__(self, gamma: torch.Tensor, beta: torch.Tensor, seed: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(STEM_IN, MID, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(MID)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(MID, OUT, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(OUT)
        generator = torch.Generator().manual_seed(seed)
        basis, _ = torch.linalg.qr(torch.randn(STEM_IN, MID, generator=generator))
        with torch.no_grad():
            self.conv1.weight.copy_(basis.t().reshape(MID, STEM_IN, 1, 1))
            self.bn1.running_mean.zero_()
            self.bn1.running_var.fill_(1.0)
            self.bn1.weight.copy_(gamma)
            self.bn1.bias.copy_(beta)
        self.eval()

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn1(self.conv1(x)))


def _calibrate(model: Stack, x: torch.Tensor) -> None:
    """Give ``bn2`` the exact population statistics of the fp32 network."""
    model.bn2.reset_running_stats()
    model.bn2.momentum = None
    model.bn2.train()
    with torch.no_grad():
        for start in range(0, x.shape[0], BATCH):
            model.bn2(model.conv2(model.features(x[start:start + BATCH])))
    model.eval()


def _channel_moments(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = y.transpose(0, 1).reshape(y.shape[1], -1)
    return flat.mean(dim=1), flat.var(dim=1)


def _synthetic_run(gamma: torch.Tensor, beta: torch.Tensor, seed: int):
    """Quantize the second convolution and compare prediction against measurement."""
    torch.manual_seed(seed)
    model = Stack(gamma, beta, seed=seed)
    x = torch.randn(IMAGES, STEM_IN, SIDE, SIDE)
    _calibrate(model, x)
    stale_mean = model.bn2.running_mean.clone()
    stale_var = model.bn2.running_var.clone()

    with torch.no_grad():
        features = torch.cat([model.features(x[i:i + BATCH])
                              for i in range(0, IMAGES, BATCH)])
    fp32_weight = model.conv2.weight.detach().clone()

    site = ConvBnSite("conv2", "bn2", CASE_A, ("bn1",), (), ACT_RELU, "bn1")
    e, s = input_moments(model, [site])["conv2"]

    residuals = quantize_with_residuals(model, passes=PASSES)
    with torch.no_grad():
        measured_mean, measured_var = _channel_moments(model.conv2(features))

    residual = residuals["conv2"]
    correction = qbn_correction(fp32_weight.reshape(OUT, -1),
                                residual.dw.reshape(OUT, -1), e, s,
                                stale_mean, stale_var, residual.group_size)
    return correction, stale_mean, stale_var, measured_mean, measured_var


def test_predicted_variance_beats_the_stale_one_when_the_model_holds():
    """The decisive synthetic check: equal gammas make ``Cov(x) = tau I`` exactly.

    The cross term section 4.5 drops is then identically zero, so the prediction
    is the whole truth and must be close to exact. This is the case the method
    is derived for; the next test is the case it is actually used in.
    """
    gamma = torch.full((MID,), 1.3)
    beta = torch.full((MID,), 0.4)
    correction, _, stale_var, _, measured_var = _synthetic_run(gamma, beta, seed=3)
    predicted_error = ((correction.running_var - measured_var).abs() / measured_var)
    stale_error = ((stale_var - measured_var).abs() / measured_var)
    assert float(predicted_error.median()) < 0.2 * float(stale_error.median())
    assert float(predicted_error.median()) < 0.005


def test_predicted_mean_beats_the_stale_one():
    """The mean line is exact given ``E[x]``, and is the larger half of the method."""
    gamma = torch.full((MID,), 1.3)
    beta = torch.full((MID,), 0.4)
    correction, stale_mean, stale_var, measured_mean, _ = _synthetic_run(
        gamma, beta, seed=3)
    units = stale_var.sqrt()
    predicted_error = (correction.running_mean - measured_mean).abs() / units
    stale_error = (stale_mean - measured_mean).abs() / units
    assert float(predicted_error.median()) < 0.1 * float(stale_error.median())


def test_predicted_variance_still_beats_the_stale_one_with_heterogeneous_channels():
    """Unequal gammas make ``Cov(x)`` diagonal but not isotropic, so the cross term lives.

    Section 8.4 says QBN then corrects the channel-averaged output variance
    rather than the per-channel one, so the aggregate is what is asserted.
    """
    gamma = torch.linspace(0.4, 2.2, MID)
    beta = torch.linspace(-0.8, 1.1, MID)
    correction, _, stale_var, _, measured_var = _synthetic_run(gamma, beta, seed=5)
    predicted_error = ((correction.running_var - measured_var).abs() / measured_var)
    stale_error = ((stale_var - measured_var).abs() / measured_var)
    assert float(predicted_error.median()) < float(stale_error.median())


def test_variance_correction_is_a_contraction_at_this_magnitude():
    """Section 3.2: ApexQuant's error is not isotropic and the output variance shrinks."""
    gamma = torch.full((MID,), 1.3)
    beta = torch.full((MID,), 0.4)
    correction, _, _, _, _ = _synthetic_run(gamma, beta, seed=3)
    assert float(correction.alpha.median()) < 0.0
    assert float(correction.d_var.median()) < 0.0


def test_harness_group_size_agrees_with_the_paper_script():
    """The duplicated group rule must not drift from the script it mirrors."""
    harness = _load_harness()
    for fan_in in range(1, 4 * MAX_GROUP + 1):
        assert harness_group_size(fan_in) == harness.group_for_fanin(fan_in), fan_in
    assert harness.MAX_GROUP == MAX_GROUP
    assert harness.MIN_FANIN_GROUP == MIN_FANIN_GROUP
    assert harness_group_size(147) is None
    assert harness_group_size(9) is None
    assert harness_group_size(FAN_IN) == 16


def test_quantize_with_residuals_reproduces_the_harness_weights():
    """Same selection, same reshape, same configuration; only ``dW`` is extra."""
    harness = _load_harness()
    reference = timm.create_model("resnet18", pretrained=False).eval()
    mirror = timm.create_model("resnet18", pretrained=False).eval()
    mirror.load_state_dict(reference.state_dict())
    harness.quantize_in_place(reference, PASSES)
    residuals = quantize_with_residuals(mirror, PASSES)
    for name, module in mirror.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim not in (2, 4):
            continue
        assert torch.equal(weight, reference.get_submodule(name).weight), name
    assert set(residuals) <= {n for n, _ in mirror.named_modules()}
    for name, residual in residuals.items():
        assert residual.dw.shape == mirror.get_submodule(name).weight.shape


def test_residual_recovers_the_fp32_weight():
    """``W = W_hat - dW`` is what the driver reconstructs, so it must be tight.

    Not bit-exact: ``dW`` is itself a float32 subtraction, and undoing a rounded
    subtraction does not always land back on the original bits. The residual is
    at float32 epsilon, five orders below the ``alpha_c`` of about -0.014 that is
    computed from it, so the recovery is exact for every purpose section 4.5 has.
    """
    model = timm.create_model("resnet18", pretrained=False).eval()
    before = {n: m.weight.detach().clone() for n, m in model.named_modules()
              if getattr(m, "weight", None) is not None and m.weight.ndim in (2, 4)}
    residuals = quantize_with_residuals(model, PASSES)
    for name, residual in residuals.items():
        recovered = model.get_submodule(name).weight.detach() - residual.dw
        error = float((recovered - before[name]).abs().max())
        assert error <= FLOAT32_ROUNDTRIP * float(before[name].abs().max()), name


@pytest.mark.parametrize("mode", [MODE_MEAN, MODE_MEAN_VAR, MODE_VAR])
def test_convnext_tiny_is_left_bit_identical(mode):
    """The negative control: no BatchNorm, so no arm may touch a single buffer."""
    model = timm.create_model("convnext_tiny", pretrained=False).eval()
    residuals = quantize_with_residuals(model, PASSES)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    reports = apply_qbn(model, residuals, mode)
    after = model.state_dict()
    assert [k for k in before if not torch.equal(before[k], after[k])] == []
    assert all(not report.corrected for report in reports.values())
    assert {report.skipped for report in reports.values()} == {
        SKIP_NO_RESIDUAL, SKIP_UNCORRECTABLE_CASE}


def test_resnet50_case_split_and_coverage():
    """32 case-A and 18 case-B convolutions are corrected; the stem stays dense."""
    model = timm.create_model("resnet50", pretrained=False).eval()
    residuals = quantize_with_residuals(model, PASSES)
    reports = apply_qbn(model, residuals, MODE_MEAN_VAR)
    corrected = [r for r in reports.values() if r.corrected]
    assert len(reports) == 53
    assert sum(r.case == CASE_A for r in corrected) == 32
    assert sum(r.case == CASE_B for r in corrected) == 18
    assert [r.conv for r in reports.values() if r.case == CASE_C] == ["conv1"]
    assert reports["conv1"].skipped == SKIP_NO_RESIDUAL
    assert sum(r.case == CASE_D and not r.corrected for r in reports.values()) == 2
    assert all(r.skipped == SKIP_NONE for r in corrected)


def test_reported_alpha_is_taken_against_the_fp32_weight():
    """A regression guard: the model already holds ``W_hat`` when the driver runs.

    Reading ``W_hat`` instead of ``W_hat - dW`` makes alpha's numerator
    ``alpha ||W||^2 + ||dW||^2``, which is near zero at every ``P``. The
    contraction term would then quietly vanish and look like a null result.
    """
    model = timm.create_model("resnet50", pretrained=False).eval()
    conv = "layer1.0.conv2"
    fp32_weight = model.get_submodule(conv).weight.detach().clone()
    residuals = quantize_with_residuals(model, PASSES)
    reports = apply_qbn(model, residuals, MODE_MEAN_VAR)
    rows = fp32_weight.shape[0]
    expected = row_alpha(fp32_weight.reshape(rows, -1),
                         residuals[conv].dw.reshape(rows, -1))
    assert torch.allclose(reports[conv].correction.alpha, expected, atol=1e-5)
    assert float(expected.median()) < -0.005


@pytest.mark.parametrize("mode,mean_moves,var_moves", [
    (MODE_MEAN, True, False),
    (MODE_VAR, False, True),
    (MODE_MEAN_VAR, True, True),
])
def test_each_arm_writes_only_its_own_buffer(mode, mean_moves, var_moves):
    """Arms 3, 4 and 5 of section 7.3 are one code path with a flag."""
    model = timm.create_model("resnet18", pretrained=False).eval()
    residuals = quantize_with_residuals(model, PASSES)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    apply_qbn(model, residuals, mode)
    after = model.state_dict()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert any(k.endswith("running_mean") for k in changed) == mean_moves
    assert any(k.endswith("running_var") for k in changed) == var_moves
    assert not any(k.endswith("weight") or k.endswith("bias") for k in changed)


def test_gamma_and_beta_are_never_touched():
    """What keeps QBN local rather than recursive, and foldable afterwards."""
    model = timm.create_model("resnet18", pretrained=False).eval()
    residuals = quantize_with_residuals(model, PASSES)
    affine = {n: (m.weight.detach().clone(), m.bias.detach().clone())
              for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)}
    apply_qbn(model, residuals, MODE_MEAN_VAR)
    for name, (gamma, beta) in affine.items():
        module = model.get_submodule(name)
        assert torch.equal(module.weight, gamma), name
        assert torch.equal(module.bias, beta), name


def test_applying_twice_is_not_the_same_as_applying_once():
    """One pass, by construction. A second pass re-anchors on corrected buffers."""
    model = timm.create_model("resnet18", pretrained=False).eval()
    residuals = quantize_with_residuals(model, PASSES)
    apply_qbn(model, residuals, MODE_MEAN_VAR)
    once = {k: v.clone() for k, v in model.state_dict().items()}
    apply_qbn(model, residuals, MODE_MEAN_VAR)
    twice = model.state_dict()
    assert any(not torch.equal(once[k], twice[k]) for k in once)


def test_efficientnet_b0_reports_its_case_d_fraction():
    """Section 8.7: publish the uncorrected count next to the accuracy."""
    model = timm.create_model("efficientnet_b0", pretrained=False).eval()
    residuals = quantize_with_residuals(model, PASSES)
    reports = apply_qbn(model, residuals, MODE_MEAN_VAR)
    assert len(reports) == 81
    assert sum(r.case == CASE_D for r in reports.values()) == 48
    assert all(not r.corrected for r in reports.values() if r.case == CASE_D)
    corrected = [r for r in reports.values() if r.corrected]
    assert len(corrected) > 0
    assert all(torch.isfinite(r.correction.d_var).all() for r in corrected)


def test_unknown_mode_fails_fast():
    """Three arms, named; a typo must not silently become a no-op."""
    model = timm.create_model("resnet18", pretrained=False).eval()
    with pytest.raises(ValueError, match="mode must be"):
        apply_qbn(model, {}, "mean-and-variance")


def test_residual_of_the_wrong_shape_fails_fast():
    """Section 8.3's layout assertion, checked before anything is written."""
    model = timm.create_model("resnet18", pretrained=False).eval()
    conv = "layer1.0.conv1"
    wrong = LayerResidual(dw=torch.zeros(1, 1, 1, 1), group_size=64)
    with pytest.raises(ValueError, match="residual is"):
        apply_qbn(model, {conv: wrong}, MODE_MEAN_VAR)


def test_a_quantized_case_c_layer_is_an_assertion_failure():
    """Section 4.4 asserts the stem is never quantized; if it is, say so loudly."""
    model = timm.create_model("resnet50", pretrained=False).eval()
    stem = model.conv1.weight
    residual = LayerResidual(dw=torch.zeros_like(stem), group_size=49)
    with pytest.raises(ValueError, match="reads the image"):
        apply_qbn(model, {"conv1": residual}, MODE_MEAN_VAR)
