"""Read the activation a timm ``BatchNormAct2d`` carries inside the BatchNorm."""

from __future__ import annotations

from torch import nn

from .activation_kind import activation_kind
from .activation_kinds import ACT_IDENTITY

_BATCHNORM_ACT = "BatchNormAct2d"
_TAIL_ATTRS = ("drop", "act")


def bundled_activation(bn: nn.Module) -> str:
    """Return the activation kind bundled into ``bn``, or ``ACT_IDENTITY``.

    timm's ``BatchNormAct2d`` subclasses ``nn.BatchNorm2d`` but also holds the
    block's activation, so treating it as a plain BatchNorm loses the
    nonlinearity -- 49 SiLUs in EfficientNet-B0, which once produced a nonsense
    0.10 percent top-1. Detection is by class name over the MRO so that this
    package does not take a hard dependency on timm.

    Any other BatchNorm subclass carrying a non-Identity ``drop`` or ``act`` tail
    is an unrecognised variant of the same trap, and raises rather than being
    read as a plain BatchNorm.

    Args:
        bn: A BatchNorm module.

    Returns:
        The activation-kind name applied to this BatchNorm's output.
    """
    if any(cls.__name__ == _BATCHNORM_ACT for cls in type(bn).__mro__):
        return activation_kind(bn.act)
    tail = [getattr(bn, attr, None) for attr in _TAIL_ATTRS]
    if any(m is not None and not isinstance(m, nn.Identity) for m in tail):
        raise TypeError(
            f"{type(bn).__name__} is not a {_BATCHNORM_ACT} but carries a "
            f"{_TAIL_ATTRS} tail; its activation cannot be read safely")
    return ACT_IDENTITY
