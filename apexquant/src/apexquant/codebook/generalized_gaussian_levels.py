"""Heavy-tailed replacement for the 1MAD generator, keeping the trellis (1MAD-p)."""

from __future__ import annotations

import math

import torch

from .symmetrize_levels import symmetrize_levels


def generalized_gaussian_levels(bits: int, shape: float = 2.0, grid: int = 40001,
                                span: float = 12.0, iters: int = 200,
                                device=None) -> torch.Tensor:
    """Unit-variance Lloyd-Max codebook for a generalized Gaussian of shape ``p``.

    QTIP's trellis is the best stage we ship, and its 1MAD generator computes
    codewords whose marginal targets a Gaussian. This function produces the
    marginal for the ``1MAD-p`` variant: the trellis is kept exactly as it is,
    which is what makes the stage stateless and table-free, and only the
    generator's target density changes to

        ``f(x) ~ exp(-|x / alpha| ** p)``,

    rescaled to unit variance. ``p = 2`` is a Gaussian and recovers 1MAD;
    ``p = 1`` is Laplace; ``p < 1`` is the hyper-Laplacian HQQ uses to model
    outlier errors. Deployed, the change is a monotone shaping map applied to the
    1MAD output -- Gaussian quantile in, generalized-Gaussian quantile out --
    which preserves every property the trellis was chosen for. Here we fit the
    resulting marginal directly, since what we need is ``rho`` and the end-to-end
    KL rather than a bit-exact generator.

    The obvious objection is that our source is compactly supported, so heavier
    tails cannot be source matching. They are not. At the deployed fan-in ``f_d``
    is already close to Gaussian and 1MAD is already well matched to it in mean
    squared error. The reason to push codeword mass outward is that the quantity
    both HQQ and SINQ find predictive of end-to-end damage is the *tail* of the
    error rather than its mean square, so this trades ``rho`` for a shorter error
    tail and is measured on both axes rather than argued.

    Args:
        bits: Codebook bit-width; ``2**bits`` levels.
        shape: Generalized-Gaussian exponent ``p``. ``2.0`` is Gaussian.
        grid: Quadrature points across the support.
        span: Half-width of the quadrature window in units of the standard
            deviation. Heavier tails need more; 12 sigma holds the ``p >= 0.5``
            mass to well past float32 resolution.
        iters: Lloyd-Max fixed-point sweeps.
        device: Optional device for the returned tensor.

    Returns:
        Sorted 1-D float32 tensor of ``2**bits`` unit-variance levels.
    """
    if shape <= 0:
        raise ValueError("shape must be > 0")

    n = 2 ** bits
    # alpha chosen so the density has unit variance: var = alpha^2 G(3/p)/G(1/p).
    lg = math.lgamma
    alpha = math.sqrt(math.exp(lg(1.0 / shape) - lg(3.0 / shape)))

    x = torch.linspace(-span, span, grid, dtype=torch.float64)
    # Work in logs: |x/alpha|**p overflows the exponential for heavy tails long
    # before the density itself is negligible.
    log_w = -((x / alpha).abs() ** shape)
    w = torch.exp(log_w - log_w.max())
    w = w / w.sum()

    cdf = torch.cumsum(w, dim=0)
    targets = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    levels = x[torch.searchsorted(cdf, targets).clamp(max=grid - 1)].clone()

    for _ in range(iters):
        boundaries = 0.5 * (levels[1:] + levels[:-1])
        cell = torch.bucketize(x, boundaries)
        num = torch.zeros(n, dtype=torch.float64).scatter_add_(0, cell, x * w)
        den = torch.zeros(n, dtype=torch.float64).scatter_add_(0, cell, w)
        levels = torch.where(den > 1e-20, num / den.clamp_min(1e-30), levels)

    return symmetrize_levels(levels).to(dtype=torch.float32, device=device)
