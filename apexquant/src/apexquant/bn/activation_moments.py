"""First and second moments of one named activation applied to a Gaussian."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .activation_kinds import (ACT_GELU, ACT_HARDSIGMOID, ACT_HARDSWISH,
                               ACT_IDENTITY, ACT_RELU, ACT_RELU6, ACT_SIGMOID,
                               ACT_SILU)
from .clipped_linear_moments import clipped_linear_moments
from .quadrature_moments import DEFAULT_NODES, quadrature_moments
from .relu_moments import relu_moments

#: ``ReLU6`` is the clipped-linear activation on this interval.
RELU6_LOW = 0.0
RELU6_HIGH = 6.0

#: Section 4.4 fixes sixteen Gauss-Hermite nodes, and for a narrow channel that
#: is right: at ``sigma = 1`` the sixteen-node second moment of SiLU is 1.5e-7
#: from a 256-node reference. It degrades fast with width, though -- 2.9e-3 for
#: GELU at ``sigma = 2`` and 3.0e-2 at ``sigma = 3`` -- and the quantity the whole
#: method is trying to measure is 0.88% of a standard deviation, so a 0.3%
#: quadrature error inside it would be a third of the signal. Wide channels
#: therefore get a larger rule. The threshold is one standard deviation rather
#: than two because that is where the measured error crosses 1e-4.
WIDE_SIGMA = 1.0
WIDE_NODES = 48

_QUADRATURE_OF_KIND = {
    ACT_SILU: F.silu,
    ACT_GELU: F.gelu,
    ACT_HARDSWISH: F.hardswish,
    ACT_SIGMOID: torch.sigmoid,
    ACT_HARDSIGMOID: F.hardsigmoid,
}


def activation_moments(activation: str, m: torch.Tensor,
                       sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(e, s)`` for ``x = activation(z)`` with ``z ~ N(m, sigma^2)``.

    This is the single dispatch point of section 4.4's case A, and case B routes
    through it too once the branch means and variances have been summed. Three
    families, in decreasing order of exactness:

    * ``identity`` is algebra, ``e = m`` and ``s = m^2 + sigma^2``.
    * ``relu`` and ``relu6`` have closed forms (the rectified and the clipped
      normal), both Monte-Carlo checked in this package's tests because the
      published printings of each contain a typo.
    * ``silu``, ``gelu``, ``hardswish``, ``sigmoid`` and ``hardsigmoid`` have no
      usable closed form for the second moment and go to Gauss-Hermite
      quadrature. EfficientNet-B0 is entirely on this path.

    ``s`` is the raw second moment ``E[x^2]``, not the variance. Section 4.5's
    ``tau`` needs ``tr(Cov x)/d`` and therefore the centred form; the subtraction
    happens in :func:`group_tau`, not here.

    Args:
        activation: One of the names in :mod:`activation_kinds`. ``image`` and
            ``unknown`` are rejected: they mark inputs the oracle does not model
            (cases C and D), and returning moments for them would let an
            uncorrectable layer be corrected by accident.
        m: Pre-activation mean, broadcast against ``sigma``.
        sigma: Pre-activation standard deviation, non-negative.

    Returns:
        Tuple ``(e, s)`` broadcast to the common shape of ``m`` and ``sigma``.

    Raises:
        ValueError: The activation has no moment model.
    """
    if activation == ACT_IDENTITY:
        return m + torch.zeros_like(sigma), m * m + sigma * sigma
    if activation == ACT_RELU:
        return relu_moments(m, sigma)
    if activation == ACT_RELU6:
        return clipped_linear_moments(m, sigma, RELU6_LOW, RELU6_HIGH)
    g = _QUADRATURE_OF_KIND.get(activation)
    if g is None:
        raise ValueError(
            f"no moment model for activation {activation!r}; cases C and D are "
            "uncorrected by construction and must not reach this function")
    nodes = WIDE_NODES if float(sigma.max()) > WIDE_SIGMA else DEFAULT_NODES
    return quadrature_moments(m, sigma, g, nodes)
