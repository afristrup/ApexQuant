"""The seven arms of section 7.3, assembled on CPU with no ImageNet in sight."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import timm
import torch
from torch import nn

from apexquant.bn.arm_model import arm_model
from apexquant.bn.qbn_arms import (ARM_FOLDED, ARM_FP32, ARM_NUMBERS,
                                          ARM_QBN_MEAN, ARM_QBN_MEAN_VAR,
                                          ARM_QBN_VAR, ARM_QUANTIZED,
                                          ARM_REESTIMATED)
from apexquant.bn.quantize_with_residuals import quantize_with_residuals

PASSES = 2
SEED = 0
BATCH = 2
SIDE = 224
CHANNELS = 3

#: Fold-then-quantize against quantize-then-fold, relative L2 on the logits.
#: With every fold constant equal, which is what an untrained model has, the two
#: orders agree at the 5e-7 section 2.1 reports. With a different positive
#: constant per row they agree only to 1.2e-3 here, because a handful of
#: nearest-codeword ties flip under the fp32 rounding of the scaled row and the
#: second residual pass makes each flip visible. Measured, not assumed.
UNIFORM_FOLD_TOL = 1e-5
VARIED_FOLD_TOL = 5e-3

EXPERIMENTS = pathlib.Path(__file__).resolve().parents[1] / "experiments"


def _load_experiment(name: str):
    """Import an experiment script by path without running it."""
    sys.path.insert(0, str(EXPERIMENTS))
    try:
        spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXPERIMENTS))


FOLD = _load_experiment("bn_stale_statistics_imagenet").fold_batchnorm


def _model(name: str = "resnet18") -> nn.Module:
    """A deterministically initialised model: two calls give identical weights."""
    torch.manual_seed(SEED)
    return timm.create_model(name, pretrained=False).eval()


def _buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: buffer.clone() for name, buffer in model.named_buffers()}


def _affine(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: module.weight.detach().clone()
            for name, module in model.named_modules()
            if isinstance(module, nn.BatchNorm2d)}


def test_two_builds_of_the_same_model_agree_bit_for_bit():
    left, right = _model(), _model()
    for a, b in zip(left.parameters(), right.parameters()):
        assert torch.equal(a, b)


def test_arm_1_is_the_untouched_fp32_model():
    reference, arm = _model(), _model()
    assert arm_model(arm, ARM_FP32, PASSES) is arm
    for a, b in zip(reference.state_dict().values(), arm.state_dict().values()):
        assert torch.equal(a, b)


def test_arm_2_is_exactly_quantize_with_residuals():
    twin = _model()
    quantize_with_residuals(twin, PASSES)
    arm = arm_model(_model(), ARM_QUANTIZED, PASSES)
    for key, value in twin.state_dict().items():
        assert torch.equal(value, arm.state_dict()[key]), key


def test_arm_2_actually_moves_the_weights():
    fp32 = _model()
    arm = arm_model(_model(), ARM_QUANTIZED, PASSES)
    assert not torch.equal(fp32.layer1[0].conv1.weight, arm.layer1[0].conv1.weight)


@pytest.mark.parametrize("arm,writes_mean,writes_var", [
    (ARM_QBN_MEAN, True, False),
    (ARM_QBN_MEAN_VAR, True, True),
    (ARM_QBN_VAR, False, True),
])
def test_the_qbn_arms_write_only_the_buffers_their_mode_allows(arm, writes_mean,
                                                               writes_var):
    quantized = _buffers(arm_model(_model(), ARM_QUANTIZED, PASSES))
    corrected = arm_model(_model(), arm, PASSES)
    moved_mean = [name for name, value in _buffers(corrected).items()
                  if name.endswith("running_mean") and not torch.equal(value, quantized[name])]
    moved_var = [name for name, value in _buffers(corrected).items()
                 if name.endswith("running_var") and not torch.equal(value, quantized[name])]
    assert bool(moved_mean) is writes_mean
    assert bool(moved_var) is writes_var


@pytest.mark.parametrize("arm", [ARM_QBN_MEAN, ARM_QBN_MEAN_VAR, ARM_QBN_VAR])
def test_the_qbn_arms_never_touch_gamma(arm):
    fp32 = _affine(_model())
    corrected = _affine(arm_model(_model(), arm, PASSES))
    for name, value in fp32.items():
        assert torch.equal(value, corrected[name]), name


def test_arm_6_re_estimates_after_quantizing_and_not_before():
    quantized = arm_model(_model(), ARM_QUANTIZED, PASSES)
    seen: list[torch.Tensor] = []
    arm = arm_model(_model(), ARM_REESTIMATED, PASSES,
                    reestimate=lambda m: seen.append(m.layer1[0].conv1.weight.clone()))
    assert len(seen) == 1
    # The model handed to the re-estimator already holds W_hat, which is the
    # whole point of arm 6: it measures the quantized network's statistics.
    assert torch.equal(seen[0], quantized.layer1[0].conv1.weight)
    assert torch.equal(arm.layer1[0].conv1.weight, quantized.layer1[0].conv1.weight)


def test_arm_7_folds_before_quantizing():
    arm = arm_model(_model(), ARM_FOLDED, PASSES, fold=FOLD)
    assert not [m for m in arm.modules() if isinstance(m, nn.BatchNorm2d)]


def _randomized(model: nn.Module) -> nn.Module:
    """Give every BatchNorm a different *positive* fold constant.

    An untrained model has ``gamma = 1`` and ``running_var = 1`` everywhere, so
    its fold constants are all one and section 2.1's identity would hold
    trivially. Section 8.6 restricts the identity to ``gamma > 0``, where the
    quantizer is sign-equivariant, so the draw is positive.
    """
    torch.manual_seed(SEED + 1)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.weight.uniform_(0.2, 2.0)
                module.bias.normal_()
                module.running_mean.normal_()
                module.running_var.uniform_(0.1, 3.0)
    return model


def _fold_order_gap(model: nn.Module, twin: nn.Module) -> float:
    """Relative L2 between folding before and after quantizing the same model."""
    folded_first = arm_model(model, ARM_FOLDED, PASSES, fold=FOLD)
    quantized_first = arm_model(twin, ARM_QUANTIZED, PASSES)
    FOLD(quantized_first)
    torch.manual_seed(SEED)
    x = torch.randn(BATCH, CHANNELS, SIDE, SIDE)
    with torch.no_grad():
        left, right = folded_first(x), quantized_first(x)
    return ((left - right).norm() / left.norm()).item()


def test_arm_7_is_the_same_function_as_folding_arm_2():
    """Section 2.1: ``fold(Q(W)) == Q(fold(W))`` for positive fold constants."""
    assert _fold_order_gap(_model(), _model()) < UNIFORM_FOLD_TOL


def test_a_different_fold_constant_per_row_costs_three_digits_of_that():
    """The same identity where the constants actually differ, which is the real case.

    An untrained model has ``gamma = 1`` and ``running_var = 1`` everywhere, so
    its fold constants are all equal and the identity holds trivially. Give each
    row its own positive constant and the two orders separate by 1.2e-3 rather
    than 5e-7: a few nearest-codeword ties flip under the fp32 rounding of the
    scaled row, and the second residual pass makes each flip visible. Section
    8.6 restricts the identity to ``gamma > 0``, where the quantizer is
    sign-equivariant, so the draw is positive; negative gamma is a further 1.8%
    on top of this and is not what this test measures.
    """
    gap = _fold_order_gap(_randomized(_model()), _randomized(_model()))
    assert UNIFORM_FOLD_TOL < gap < VARIED_FOLD_TOL


@pytest.mark.parametrize("arm", [ARM_QBN_MEAN, ARM_QBN_MEAN_VAR, ARM_QBN_VAR])
def test_convnext_tiny_arms_are_bit_identical_to_arm_2(arm):
    """The negative control of section 7.3: no BatchNorm, so nothing may move."""
    quantized = arm_model(_model("convnext_tiny"), ARM_QUANTIZED, PASSES)
    corrected = arm_model(_model("convnext_tiny"), arm, PASSES)
    torch.manual_seed(SEED)
    x = torch.randn(BATCH, CHANNELS, SIDE, SIDE)
    with torch.no_grad():
        assert torch.equal(quantized(x), corrected(x))


def test_an_unknown_arm_raises():
    with pytest.raises(ValueError, match="arm must be one of"):
        arm_model(_model(), max(ARM_NUMBERS) + 1, PASSES)


def test_a_pass_count_below_one_raises_on_every_arm_that_quantizes():
    for arm in ARM_NUMBERS:
        if arm == ARM_FP32:
            continue
        with pytest.raises(ValueError, match="passes"):
            arm_model(_model(), arm, 0, fold=FOLD, reestimate=lambda m: None)


def test_the_folding_arm_refuses_to_run_without_a_folder():
    with pytest.raises(ValueError, match="fold"):
        arm_model(_model(), ARM_FOLDED, PASSES)


def test_the_re_estimating_arm_refuses_to_run_without_images():
    with pytest.raises(ValueError, match="reestimate"):
        arm_model(_model(), ARM_REESTIMATED, PASSES)
