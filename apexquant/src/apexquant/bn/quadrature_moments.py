"""Gauss-Hermite moments of an arbitrary activation of a Gaussian."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

#: Section 4.4 fixes 16 nodes and claims twelve already give relative error
#: below 1e-5 on both moments. Measured, that claim is optimistic: at ``m = 0``,
#: ``sigma = 1`` twelve nodes put GELU's second moment 1.2e-4 out, and sixteen
#: bring it to 6.7e-6. The default is therefore load bearing, not margin. Above
#: ``sigma`` of about two, neither count suffices and the caller must raise it;
#: see ``tests/test_qbn_quadrature_moments.py``.
DEFAULT_NODES = 16

#: ``hermgauss`` integrates against ``exp(-t^2)``, so the substitution is
#: ``z = m + sqrt(2)*sigma*t`` and the weights carry a ``pi^{-1/2}``.
_SQRT_TWO = math.sqrt(2.0)
_SQRT_PI = math.sqrt(math.pi)
_INV_SQRT_PI = 1.0 / _SQRT_PI

#: ``numpy``'s ``hermgauss`` overflows its Newton iteration somewhere above 256
#: nodes and returns ``nan`` silently. The weights of an exact rule sum to
#: ``sqrt(pi)``, so checking that catches the breakdown for the price of one
#: reduction.
_WEIGHT_SUM_TOLERANCE = 1e-9


def quadrature_moments(m: torch.Tensor, sigma: torch.Tensor,
                       g: Callable[[torch.Tensor], torch.Tensor],
                       nodes: int = DEFAULT_NODES) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(e, s)`` for ``x = g(z)`` with ``z ~ N(m, sigma^2)``, by quadrature.

    This is the path for the activations with no usable closed form for the
    second moment: SiLU, GELU, Hardswish. ``E[silu^2] = E[z^2 sigmoid(z)^2]`` is
    a bivariate normal CDF object, and the probit substitution
    ``sigmoid(z) ~ Phi(a z)`` errs by up to 0.05 absolute on the first moment
    alone, so neither shortcut is available. EfficientNet-B0 must take this
    path, since it contains SiLU and Sigmoid and no ReLU at all.

    With ``(t_q, w_q)`` the ``n``-node Gauss-Hermite rule for
    ``integral exp(-t^2)``,

        e = pi^{-1/2} * sum_q w_q * g(m + sqrt(2)*sigma*t_q)
        s = pi^{-1/2} * sum_q w_q * g(m + sqrt(2)*sigma*t_q)^2

    ``sigma = 0`` needs no special case here: every node collapses onto ``m``
    and the weights sum to ``sqrt(pi)``, so ``e = g(m)`` and ``s = g(m)^2``
    fall out of the same expression.

    Args:
        m: Pre-activation mean, any shape, broadcast against ``sigma``.
        sigma: Pre-activation standard deviation, non-negative.
        g: The activation, applied elementwise to a tensor of shape
            ``broadcast(m, sigma) + (nodes,)``. It must be a pure elementwise
            map; a ``g`` that mixes entries silently integrates the wrong thing.
        nodes: Number of quadrature nodes, at least one.

    Returns:
        Tuple ``(e, s)`` with ``e = E[g(z)]`` and ``s = E[g(z)^2]``, broadcast
        to the common shape of ``m`` and ``sigma``, in the promoted input dtype.
    """
    if nodes < 1:
        raise ValueError(f"nodes must be at least 1, got {nodes}")
    if bool((sigma < 0.0).any()):
        raise ValueError("sigma must be non-negative; pass |gamma|, not gamma")

    dtype = torch.promote_types(m.dtype, sigma.dtype)
    m64 = m.to(torch.float64)
    sigma64 = sigma.to(torch.float64)

    abscissa, weights = np.polynomial.hermite.hermgauss(nodes)
    if not np.isfinite(weights).all() or not np.isfinite(abscissa).all():
        raise ValueError(f"hermgauss({nodes}) returned a non-finite rule")
    if abs(weights.sum() / _SQRT_PI - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            f"hermgauss({nodes}) weights sum to {weights.sum()}, not sqrt(pi); "
            "the rule has broken down and the moments would be silently wrong"
        )
    t = torch.as_tensor(abscissa, dtype=torch.float64, device=m64.device)
    w = torch.as_tensor(weights, dtype=torch.float64, device=m64.device)

    m64, sigma64 = torch.broadcast_tensors(m64, sigma64)
    z = m64.unsqueeze(-1) + _SQRT_TWO * sigma64.unsqueeze(-1) * t
    values = g(z)
    if values.shape != z.shape:
        raise ValueError(
            f"g must be elementwise: it mapped {tuple(z.shape)} to "
            f"{tuple(values.shape)}"
        )

    e = _INV_SQRT_PI * (values * w).sum(dim=-1)
    s = _INV_SQRT_PI * (values * values * w).sum(dim=-1)
    return e.to(dtype), s.to(dtype)
