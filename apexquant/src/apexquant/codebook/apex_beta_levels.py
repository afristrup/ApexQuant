from __future__ import annotations

import torch

from .symmetrize_levels import symmetrize_levels


def apex_beta_levels(bits: int, fan_in: int, grid: int = 40001, iters: int = 200,
                     device=None) -> torch.Tensor:
    """Lloyd-Max codebook for the post-rotation coordinate density ``f_d``.

    ApexQuant observes that after row L2-normalization and a
    random orthogonal rotation, each weight coordinate is (approximately) a
    coordinate of a uniform point on the sphere ``S^{d-1}``, whose marginal is

        f_d(u) = C_d * (1 - u**2) ** ((d - 3) / 2),   u in [-1, 1]

    i.e. ``(u + 1) / 2 ~ Beta((d-1)/2, (d-1)/2)`` with variance ``1/d``. The
    MSE-optimal scalar quantizer for a known density is its Lloyd-Max codebook,
    which we solve here on a dense grid (data-free, deterministic). This *is* the
    only step where ApexQuant differs from QuaRot/TurboQuant, and it is what makes
    2-bit weight quantization viable on large-fan-in layers.

    Two limits are worth noting and are exercised by the tests:
        * ``d -> inf``: ``f_d`` scaled by ``sqrt(d)`` approaches ``N(0, 1)``, so
          ``apex_beta_levels(b, d) * sqrt(d) -> lloyd_max_levels(b)``.
        * ``d = 3``: the exponent is 0, ``f_3`` is uniform on ``[-1, 1]``, and the
          matched codebook degenerates to the uniform grid. This is ApexQuant's
          documented low-fan-in boundary (Proposition 1) -- the regime where
          SpectralQuant's data-aware rotation is needed instead.

    Args:
        bits: Codebook bit-width; ``2**bits`` levels.
        fan_in: The dimension ``d`` the coordinate came from (group/row size).
        grid: Number of quadrature points on ``[-1, 1]``.
        iters: Lloyd-Max fixed-point iterations (``f_d`` is log-concave for
            ``d >= 3`` so the iteration has a unique optimum).
        device: Optional device for the returned tensor.

    Returns:
        Sorted 1-D float32 tensor of ``2**bits`` reconstruction levels in
        ``[-1, 1]``.
    """
    if fan_in < 3:
        raise ValueError("fan_in must be >= 3 for the spherical marginal to hold")

    n = 2 ** bits
    u = torch.linspace(-1.0, 1.0, grid, dtype=torch.float64)

    # Density weights on the grid (unnormalized is fine for centroids/argmin).
    exponent = (fan_in - 3) / 2.0
    w = (1.0 - u * u).clamp_min(0.0) ** exponent
    w = w / w.sum()

    # Initialize levels at density quantiles so no cell starts empty.
    cdf = torch.cumsum(w, dim=0)
    targets = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    levels = u[torch.searchsorted(cdf, targets).clamp(max=grid - 1)].clone()

    for _ in range(iters):
        # Cell boundaries are midpoints; assign each grid point to a cell.
        boundaries = 0.5 * (levels[1:] + levels[:-1])
        cell = torch.bucketize(u, boundaries)              # 0 .. n-1
        num = torch.zeros(n, dtype=torch.float64).scatter_add_(0, cell, u * w)
        den = torch.zeros(n, dtype=torch.float64).scatter_add_(0, cell, w)
        # Centroid of each cell; keep the old level for any (rare) empty cell.
        moved = num / den.clamp_min(1e-30)
        levels = torch.where(den > 1e-20, moved, levels)

    return symmetrize_levels(levels).to(dtype=torch.float32, device=device)
