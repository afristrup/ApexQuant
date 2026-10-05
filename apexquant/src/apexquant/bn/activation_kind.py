"""Name the activation a module applies, for the input-moment oracle."""

from __future__ import annotations

from torch import nn

from .activation_kinds import (ACT_GELU, ACT_HARDSIGMOID, ACT_HARDSWISH,
                               ACT_IDENTITY, ACT_RELU, ACT_RELU6, ACT_SIGMOID,
                               ACT_SILU, ACT_UNKNOWN)

_KIND_OF_TYPE = {
    nn.Identity: ACT_IDENTITY,
    nn.ReLU: ACT_RELU,
    nn.ReLU6: ACT_RELU6,
    nn.SiLU: ACT_SILU,
    nn.GELU: ACT_GELU,
    nn.Sigmoid: ACT_SIGMOID,
    nn.Hardswish: ACT_HARDSWISH,
    nn.Hardsigmoid: ACT_HARDSIGMOID,
}


def activation_kind(module: nn.Module | None) -> str:
    """Return the activation-kind name of ``module``.

    Exact type lookup rather than ``isinstance``: a subclass of ``nn.ReLU`` that
    clips somewhere else must not be silently treated as a ReLU, because the
    closed-form moments of section 4.4 would then be wrong. Anything unrecognised
    comes back as ``ACT_UNKNOWN``, which the caller is expected to route to case D.

    Args:
        module: The activation module, or None for "no activation applied".

    Returns:
        One of the names in :mod:`activation_kinds`.
    """
    if module is None:
        return ACT_IDENTITY
    return _KIND_OF_TYPE.get(type(module), ACT_UNKNOWN)
