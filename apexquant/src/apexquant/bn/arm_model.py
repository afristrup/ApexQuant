"""Prepare one model for one arm of the section 7.3 accuracy experiment."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from .apply_qbn import apply_qbn
from .qbn_arms import (ARM_FOLDED, ARM_FP32, ARM_MODES, ARM_NUMBERS,
                       ARM_QUANTIZED, ARM_REESTIMATED)
from .quantize_with_residuals import quantize_with_residuals

MIN_PASSES = 1


def arm_model(model: nn.Module, arm: int, passes: int, *,
              fold: Callable[[nn.Module], object] | None = None,
              reestimate: Callable[[nn.Module], object] | None = None
              ) -> nn.Module:
    """Quantize and correct ``model`` in place for ``arm``, and hand it back.

    One cell of the table is one fresh model, which is why this takes a model
    rather than building one: each arm quantizes its own copy, so no arm can
    inherit a buffer another arm edited, and any cell can be run alone on a
    resumed run.

    The order of section 4.2 is forced in both directions and is the only thing
    of substance here. Arms 3 to 5 quantize first, because the correction needs
    ``dW``, and correct before anything is folded, because the correction needs
    the *unfolded* ``(gamma, beta)`` of the preceding BatchNorm. Arm 7 is the
    opposite order on purpose: it folds *first*, so that no running statistic
    survives to be stale, and section 2.1 predicts it must come out equal to arm
    2 because scaling a weight row by a positive constant leaves every codeword
    unchanged. A material difference there is a fault in the harness, not a
    result.

    ``fold`` and ``reestimate`` are injected rather than imported because both
    live in the experiment scripts the paper's numbers come from, which are not
    to be edited or duplicated. They are called for their effect on the model;
    their return values are ignored.

    Args:
        model: A freshly built fp32 model in eval mode. Modified in place.
        arm: Arm number, one of :data:`ARM_NUMBERS`.
        passes: Residual passes ``P``, at least :data:`MIN_PASSES`. Ignored by
            the fp32 arm, which does not quantize.
        fold: Folds every Conv2d-BatchNorm2d pair into the convolution, in
            place. Required by arm 7 and unused by every other arm.
        reestimate: Recomputes the running statistics from unlabelled images, in
            place. Required by arm 6 and unused by every other arm. Arm 6 is the
            only arm that is not data-free.

    Returns:
        The same module, prepared for the arm.

    Raises:
        ValueError: Unknown arm, a pass count below :data:`MIN_PASSES` on an arm
            that quantizes, or a missing ``fold`` / ``reestimate`` callable.
    """
    if arm not in ARM_NUMBERS:
        raise ValueError(f"arm must be one of {ARM_NUMBERS}, got {arm!r}")
    if arm == ARM_FP32:
        return model
    if passes < MIN_PASSES:
        raise ValueError(f"arm {arm} quantizes and needs passes >= {MIN_PASSES}, "
                         f"got {passes!r}")
    if arm == ARM_FOLDED:
        if fold is None:
            raise ValueError("arm 7 folds BatchNorm before quantizing and needs a "
                             "`fold` callable")
        fold(model)
        quantize_with_residuals(model, passes)
        return model
    residuals = quantize_with_residuals(model, passes)
    if arm == ARM_QUANTIZED:
        return model
    if arm == ARM_REESTIMATED:
        if reestimate is None:
            raise ValueError("arm 6 re-estimates BatchNorm on images and needs a "
                             "`reestimate` callable")
        reestimate(model)
        return model
    apply_qbn(model, residuals, ARM_MODES[arm])
    return model
