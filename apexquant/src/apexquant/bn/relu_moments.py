"""First and second moments of ``ReLU`` applied to a Gaussian, per channel."""

from __future__ import annotations

import math

import torch

#: ``phi(t) = exp(-t^2 / 2) / sqrt(2 pi)``.
_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)

#: Stand-in divisor for the deterministic channels. ``sigma = 0`` is a real
#: configuration (it is exactly what ``gamma = 0`` produces), so the ratio
#: ``r = m / sigma`` is computed against this placeholder and the result is
#: then discarded by a ``where``. Any strictly positive value does; one keeps
#: the intermediate arithmetic in range.
_PLACEHOLDER_SIGMA = 1.0


def relu_moments(m: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(e, s)`` for ``x = ReLU(z)`` with ``z ~ N(m, sigma^2)``.

    With ``r = m / sigma``, ``Phi`` the standard normal CDF and ``phi`` its PDF,

        e = m * Phi(r) + sigma * phi(r)
        s = (m^2 + sigma^2) * Phi(r) + m * sigma * phi(r)

    ``e`` is Nagel et al. 2019 Eq. 19. ``s`` is the rectified-Gaussian second
    moment, re-derived rather than copied: their Appendix C prints a variance
    that under-reports, giving 0.261 at ``m = 0``, ``sigma = 1`` against a true
    0.341, and Gast and Roth Eq. 10b carries the same class of typo, ``(mu + v)``
    where it must be ``(mu^2 + v)``. Both forms are checked against Monte Carlo
    in ``tests/test_qbn_relu_moments.py``, which is the only reason to trust
    this one.

    Note that ``s`` is the raw second moment ``E[x^2]``, not the variance. The
    variance is ``s - e^2``, and it is the caller's job to subtract, because the
    correction of section 4.5 needs the second moment for the mean line and the
    centred moment for ``tau``, and mixing them up inflates ``tau`` by half.

    ``sigma = 0`` is handled exactly and without a division: the channel is
    deterministic, so ``e = ReLU(m)`` and ``s = ReLU(m)^2``.

    Args:
        m: Pre-activation mean, any shape, broadcast against ``sigma``.
        sigma: Pre-activation standard deviation, non-negative. Negative
            entries are a caller bug -- most likely ``gamma`` used in place of
            ``|gamma|`` -- and are rejected.

    Returns:
        Tuple ``(e, s)`` with ``e = E[x]`` and ``s = E[x^2]``, broadcast to the
        common shape of ``m`` and ``sigma``, in the promoted input dtype.
    """
    if bool((sigma < 0.0).any()):
        raise ValueError("sigma must be non-negative; pass |gamma|, not gamma")

    dtype = torch.promote_types(m.dtype, sigma.dtype)
    m64 = m.to(torch.float64)
    sigma64 = sigma.to(torch.float64)

    positive = sigma64 > 0.0
    safe_sigma = torch.where(positive, sigma64, torch.full_like(sigma64, _PLACEHOLDER_SIGMA))
    r = m64 / safe_sigma
    cdf = torch.special.ndtr(r)
    pdf = torch.exp(-0.5 * r * r) / _SQRT_TWO_PI

    e = m64 * cdf + sigma64 * pdf
    s = (m64 * m64 + sigma64 * sigma64) * cdf + m64 * sigma64 * pdf

    rectified = torch.clamp(m64, min=0.0)
    e = torch.where(positive, e, rectified)
    s = torch.where(positive, s, rectified * rectified)
    return e.to(dtype), s.to(dtype)
