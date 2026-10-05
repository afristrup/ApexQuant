"""E8 lattice nearest-point quantizer (structured 8-D codebook, no training)."""

from __future__ import annotations

import torch


def _nearest_dn(x: torch.Tensor) -> torch.Tensor:
    """Nearest point of the D_n lattice (integers with even coordinate sum)."""
    a = torch.round(x)
    # If the coordinate sum is odd, flip the single worst-rounded coordinate the
    # other way -- the classic Conway-Sloane D_n decoder.
    diff = x - a
    k = diff.abs().argmax(dim=-1, keepdim=True)
    flip = torch.where(torch.gather(diff, -1, k) >= 0, 1.0, -1.0)
    a_flip = a.scatter(-1, k, torch.gather(a, -1, k) + flip)
    odd = (a.sum(dim=-1) % 2 != 0).unsqueeze(-1)
    return torch.where(odd, a_flip, a)


def e8_lattice_quantize(x: torch.Tensor) -> torch.Tensor:
    """Quantize the last-axis 8-vectors of ``x`` to the nearest E8 lattice point.

    E8 is the union ``D8 ∪ (D8 + ½)`` and is the densest lattice / best 8-D
    quantizer, with normalized second moment ``G(E8) ≈ 0.0717`` versus the scalar
    ``1/12 ≈ 0.0833`` -- ~14% lower MSE at the *same rate* (E8 is unimodular, so
    at unit step it has the same point density as ``Z^8``). This is the training-
    free way to scale the codebook dimension past small learned product codebooks.

    Args:
        x: tensor with last dimension 8.

    Returns:
        Nearest E8 points, same shape as ``x``.
    """
    if x.shape[-1] != 8:
        raise ValueError("E8 quantization requires a last dimension of 8")
    a = _nearest_dn(x)
    b = _nearest_dn(x - 0.5) + 0.5
    closer_a = (x - a).pow(2).sum(-1, keepdim=True) <= (x - b).pow(2).sum(-1, keepdim=True)
    return torch.where(closer_a, a, b)
