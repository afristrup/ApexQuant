"""Density-matched codebook fitted under an l_p objective instead of MSE (HQQ)."""

from __future__ import annotations

import functools

import torch

from .symmetrize_levels import symmetrize_levels


# The fit is a deterministic quadrature over a density fixed by (bits, fan_in,
# p): the same call from every layer of a model returns the same 2**bits numbers.
# Recomputing it per layer made the generalized-Lloyd sweeps, not the
# quantization, the cost of a whole-model run. The result is only ever read
# through an index, never mutated, so sharing one tensor is safe.
@functools.lru_cache(maxsize=32)
def lp_levels(bits: int, fan_in: int, p: float = 0.7, grid: int = 40001,
              iters: int = 200, irls_iters: int = 12, eps: float = 1e-6,
              device=None) -> torch.Tensor:
    """Generalized Lloyd codebook for ``f_d`` under ``E|t - q(t)|**p``.

    HQQ (Badri & Shaji, 2023) argues that squared error is the wrong loss for
    quantization because the damage is dominated by a few large errors, and
    rounds under a sparsity-promoting ``l_p`` norm at ``p = 0.7`` instead, which
    ``more accurately captures the heavy-tailed nature of outlier errors``. Our
    recursion is built on squared error throughout, so the honest test of that
    claim is to change *only* the stage codebook and leave the recursion alone.
    That is what this function produces.

    The geometric decay law is untouched by the swap: it holds for whatever
    normalized distortion ``rho**2`` the stage happens to attain, so an ``l_p``
    stage simply moves ``rho``. It moves it *up*, necessarily, since the MSE
    optimum is the Lloyd-Max codebook of ``apex_beta_levels`` by construction.
    The question this codebook exists to answer is whether KL falls anyway, and
    it is the one place where a stage can beat our own design chart, which ranks
    stages by ``rho`` and therefore cannot rank this one.

    Two structural notes on the fit:

    * The decision boundaries stay at the midpoints. ``|x - a|**p = |x - b|**p``
      iff ``|x - a| = |x - b|`` for any ``p > 0``, so only the centroid step
      changes, not the partition step.
    * The centroid of a cell under ``l_p`` has no closed form for ``p < 1``, and
      autograd is no help there either -- HQQ notes plain gradient descent
      ``fails with p < 1, which is what we actually use``. We use the same escape
      they do, an iteratively reweighted (half-quadratic) inner solve
      ``c <- sum(w |u - c|**(p-2) u) / sum(w |u - c|**(p-2))``, which is the
      closed-form minimizer of the majorizing quadratic.

    Args:
        bits: Codebook bit-width; ``2**bits`` levels.
        fan_in: Dimension ``d`` the coordinate came from, as in
            :func:`apex_beta_levels`; the source density is ``f_d``.
        p: Norm exponent. ``p = 2`` recovers the MSE codebook exactly, ``0.7`` is
            HQQ's value.
        grid: Quadrature points on ``[-1, 1]``.
        iters: Outer generalized-Lloyd sweeps.
        irls_iters: Inner reweighting steps per cell per sweep.
        eps: Floor on ``|u - c|`` inside the reweighting, which is what keeps the
            ``p < 2`` weights finite at the level itself.
        device: Optional device for the returned tensor.

    Returns:
        Sorted 1-D float32 tensor of ``2**bits`` levels in ``[-1, 1]``.
    """
    if fan_in < 3:
        raise ValueError("fan_in must be >= 3 for the spherical marginal to hold")
    if p <= 0:
        raise ValueError("p must be > 0")

    n = 2 ** bits
    u = torch.linspace(-1.0, 1.0, grid, dtype=torch.float64)

    exponent = (fan_in - 3) / 2.0
    w = (1.0 - u * u).clamp_min(0.0) ** exponent
    w = w / w.sum()

    cdf = torch.cumsum(w, dim=0)
    targets = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    levels = u[torch.searchsorted(cdf, targets).clamp(max=grid - 1)].clone()

    for _ in range(iters):
        boundaries = 0.5 * (levels[1:] + levels[:-1])
        cell = torch.bucketize(u, boundaries)                     # 0 .. n-1

        # Half-quadratic inner solve, run on all cells at once. The weight
        # |u - c|**(p-2) is the majorizer's curvature; for p = 2 it is constant
        # and one step reproduces the ordinary centroid.
        c = levels.clone()
        for _ in range(irls_iters):
            dist = (u - c[cell]).abs().clamp_min(eps)
            iw = w * dist ** (p - 2.0)
            num = torch.zeros(n, dtype=torch.float64).scatter_add_(0, cell, iw * u)
            den = torch.zeros(n, dtype=torch.float64).scatter_add_(0, cell, iw)
            c = torch.where(den > 1e-30, num / den.clamp_min(1e-30), c)
        levels = c

    return symmetrize_levels(levels.sort().values).to(dtype=torch.float32,
                                                      device=device)
