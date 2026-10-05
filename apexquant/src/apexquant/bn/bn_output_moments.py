"""Per-channel output mean and standard deviation of a BatchNorm layer."""

from __future__ import annotations

import torch
from torch import nn

#: Section 4.4 Case A: the pre-activation of the *next* layer is modelled as
#: ``N(beta, gamma^2 v / (v + eps))``. Nothing here is a tunable; the two
#: constants below only name the neutral affine parameters of a
#: ``affine=False`` BatchNorm, which is an untrained identity scale and shift.
_UNIT_GAMMA = 1.0
_ZERO_BETA = 0.0


def bn_output_moments(bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(m, sigma)``, the output moments of ``bn``, one entry per channel.

    These are the moments of what BatchNorm *emits*, not of what it consumes.
    ``running_mean`` and ``running_var`` describe the pre-normalization
    convolution output and are the anchor the variance correction edits;
    ``(beta, gamma)`` describe the normalized output and are what supply the
    next layer's input moments. Conflating the two pairs is the obvious way to
    get the correction wrong, so they live in different functions.

    ``sigma = |gamma| * sqrt(v / (v + eps))`` and the absolute value is not
    cosmetic. 103 of ResNet-50's 26,560 BatchNorm channels and 392 of
    EfficientNet-B0's 21,008 carry ``gamma < 0``, and every published form of
    this formula -- DFQ Eq. 19, the Nagel white paper Eq. 28 -- uses ``gamma``
    itself as a standard deviation and is wrong as literally written for those
    channels. A standard deviation is non-negative by definition, and the
    downstream rectified-Gaussian moments are not symmetric in its sign, so the
    error does not cancel.

    ``gamma = 0`` is a real configuration, not a degenerate one: it gives
    ``sigma = 0`` and a deterministic channel, which the moment functions in
    this package handle without dividing by zero.

    Args:
        bn: Any ``nn.BatchNorm2d``, including timm's ``BatchNormAct2d`` and
            other subclasses. It must be tracking running statistics; a module
            with ``running_var is None`` carries no anchor and is rejected
            rather than silently given a unit variance.

    Returns:
        Tuple ``(m, sigma)`` of detached tensors of shape ``(num_features,)``,
        in the dtype of ``running_var``. ``m`` is ``beta``; ``sigma`` is the
        output standard deviation and is non-negative everywhere.
    """
    var = getattr(bn, "running_var", None)
    if var is None:
        raise ValueError(
            "bn_output_moments needs running_var; this module does not track "
            "running statistics and has no anchor to read"
        )
    eps = float(bn.eps)
    if eps < 0.0:
        raise ValueError(f"bn.eps must be non-negative, got {eps}")

    var = var.detach()
    if bool((var < 0.0).any()):
        raise ValueError("running_var has negative entries; the buffer is corrupt")

    gamma = bn.weight.detach() if bn.weight is not None else torch.full_like(var, _UNIT_GAMMA)
    beta = bn.bias.detach() if bn.bias is not None else torch.full_like(var, _ZERO_BETA)

    denom = var + eps
    if bool((denom <= 0.0).any()):
        raise ValueError(
            "running_var + eps has a non-positive entry; BatchNorm would divide "
            "by zero on this channel, so its output moments are undefined"
        )

    sigma = gamma.abs() * torch.sqrt(var / denom)
    return beta.clone(), sigma
