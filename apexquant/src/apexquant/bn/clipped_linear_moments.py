"""First and second moments of a clipped-linear activation of a Gaussian."""

from __future__ import annotations

import math

import torch

#: ``phi(t) = exp(-t^2 / 2) / sqrt(2 pi)``.
_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)

#: Stand-in divisor for deterministic channels; see ``relu_moments``.
_PLACEHOLDER_SIGMA = 1.0


def clipped_linear_moments(m: torch.Tensor, sigma: torch.Tensor, a: float,
                           b: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(e, s)`` for ``x = clip(z, a, b)`` with ``z ~ N(m, sigma^2)``.

    With ``A = (a - m) / sigma`` and ``B = (b - m) / sigma``, section 4.4:

        e = a*Phi(A) + sigma*(phi(A) - phi(B)) + m*(Phi(B) - Phi(A))
            + b*(1 - Phi(B))
        s = a^2*Phi(A) + (m^2 + sigma^2)*(Phi(B) - Phi(A))
            + sigma*((a + m)*phi(A) - (b + m)*phi(B)) + b^2*(1 - Phi(B))

    The asymmetry between the two ``sigma`` terms in ``s`` is real and is where
    the published typos live: the truncated second moment contributes
    ``sigma*((a - m)*phi(A) - (b - m)*phi(B))`` and the cross term contributes a
    further ``2*m*sigma*(phi(A) - phi(B))``, and the two sum to ``(a + m)`` and
    ``(b + m)``. Copying a printed form instead of adding those two up is how
    DFQ's Appendix C ends at 0.261 where the truth is 0.341.

    This generalizes ``relu_moments`` (``a = 0``, ``b = +inf``) but does not
    replace it: ``b = inf`` makes the term ``b*(1 - Phi(B))`` an ``inf * 0``, so
    the rectifier keeps its own closed form.

    ``sigma = 0`` is handled exactly and without a division: the channel is
    deterministic, so ``e = clip(m, a, b)`` and ``s = clip(m, a, b)^2``.

    Args:
        m: Pre-activation mean, any shape, broadcast against ``sigma``.
        sigma: Pre-activation standard deviation, non-negative.
        a: Lower clipping level, finite.
        b: Upper clipping level, finite and strictly greater than ``a``.

    Returns:
        Tuple ``(e, s)`` with ``e = E[x]`` and ``s = E[x^2]``, broadcast to the
        common shape of ``m`` and ``sigma``, in the promoted input dtype.
    """
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError(
            f"clipping levels must be finite, got a={a}, b={b}; an unbounded "
            "upper level is relu_moments, not this function"
        )
    if not b > a:
        raise ValueError(f"need b > a, got a={a}, b={b}")
    if bool((sigma < 0.0).any()):
        raise ValueError("sigma must be non-negative; pass |gamma|, not gamma")

    dtype = torch.promote_types(m.dtype, sigma.dtype)
    m64 = m.to(torch.float64)
    sigma64 = sigma.to(torch.float64)

    positive = sigma64 > 0.0
    safe_sigma = torch.where(positive, sigma64, torch.full_like(sigma64, _PLACEHOLDER_SIGMA))
    lower = (a - m64) / safe_sigma
    upper = (b - m64) / safe_sigma
    cdf_a = torch.special.ndtr(lower)
    cdf_b = torch.special.ndtr(upper)
    pdf_a = torch.exp(-0.5 * lower * lower) / _SQRT_TWO_PI
    pdf_b = torch.exp(-0.5 * upper * upper) / _SQRT_TWO_PI

    inside = cdf_b - cdf_a
    above = 1.0 - cdf_b
    e = a * cdf_a + sigma64 * (pdf_a - pdf_b) + m64 * inside + b * above
    s = (a * a * cdf_a
         + (m64 * m64 + sigma64 * sigma64) * inside
         + sigma64 * ((a + m64) * pdf_a - (b + m64) * pdf_b)
         + b * b * above)

    clipped = torch.clamp(m64, min=a, max=b)
    e = torch.where(positive, e, clipped)
    s = torch.where(positive, s, clipped * clipped)
    return e.to(dtype), s.to(dtype)
