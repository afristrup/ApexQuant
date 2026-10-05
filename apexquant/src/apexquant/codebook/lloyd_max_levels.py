"""Lloyd-Max optimal scalar codebook for a unit-variance Gaussian source."""

from __future__ import annotations

import math

import torch

from .symmetrize_levels import symmetrize_levels


def lloyd_max_levels(bits: int, iters: int = 200, device=None) -> torch.Tensor:
    """Return the ``2**bits`` Lloyd-Max reconstruction levels for ``N(0, 1)``.

    After random rotation and per-group normalization, weight coordinates are
    very close to i.i.d. standard normal, so the MSE-optimal scalar quantizer is
    the Lloyd-Max quantizer of a unit Gaussian. We solve it here with the
    classic fixed-point iteration (deterministic, no data needed):

        * decision boundaries sit halfway between neighbouring levels;
        * each level moves to the centroid of its Gaussian-weighted cell,
          which for a truncated normal is ``(phi(a) - phi(b)) / (Phi(b) - Phi(a))``.

    Args:
        bits: Codebook bit-width; the codebook has ``2**bits`` entries.
        iters: Fixed-point iterations. 200 is far past convergence for <=8 bits.
        device: Optional target device for the returned tensor.

    Returns:
        Sorted 1-D float32 tensor of length ``2**bits``.
    """
    n = 2 ** bits
    # Symmetric initial guess spanning the bulk of the Gaussian.
    levels = torch.linspace(-2.5, 2.5, n, dtype=torch.float64)

    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

    def phi(x: torch.Tensor) -> torch.Tensor:  # standard-normal pdf
        return inv_sqrt_2pi * torch.exp(-0.5 * x * x)

    def Phi(x: torch.Tensor) -> torch.Tensor:  # standard-normal cdf
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    neg_inf = torch.tensor([-1e6], dtype=torch.float64)
    pos_inf = torch.tensor([1e6], dtype=torch.float64)

    for _ in range(iters):
        # Boundaries: -inf, midpoints, +inf.
        mids = 0.5 * (levels[1:] + levels[:-1])
        lo = torch.cat([neg_inf, mids])
        hi = torch.cat([mids, pos_inf])
        mass = (Phi(hi) - Phi(lo)).clamp_min(1e-12)
        levels = (phi(lo) - phi(hi)) / mass  # centroid of each Gaussian cell

    return symmetrize_levels(levels).to(dtype=torch.float32, device=device)
