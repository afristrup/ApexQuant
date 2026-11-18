"""DASHQuant's base quantizer (Algorithm 1) on one weight matrix, reconstructed densely."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from apexquant.rotation.inverse_random_rotation import inverse_random_rotation
from apexquant.rotation.random_rotation import random_rotation


@torch.no_grad()
def reconstruct_dash(weight: torch.Tensor, bits: int, group: int, seed: int) -> torch.Tensor:
    """One randomized Hadamard transform, a shared dither, and the dithered quantile codebook.

    Per group: store the norm, rotate the unit vector by ``H diag(signs)``, scale by
    ``sqrt(d)``, and round each coordinate to the bucket of ``F(t) = Phi(t / sqrt 3)``
    whose edges are ``h_0 = 0``, ``h_j = (j + U) / B``, ``h_B = 1``, reconstructing at
    ``F^{-1}`` of the bucket midpoint. ``U`` is uniform per group and, like the signs,
    derived from ``seed``, so the decoder needs no extra bits. Reimplemented from
    Feng et al. (2026); no data is read.
    """
    out_f, in_f = weight.shape
    pad = (-in_f) % group
    w = F.pad(weight.float(), (0, pad)).view(out_f, -1, group)
    norm = w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    z = math.sqrt(group) * random_rotation(w / norm, seed)
    levels = 2 ** bits
    gen = torch.Generator(device=w.device).manual_seed(seed + 7919)
    u = torch.rand(out_f, w.shape[1], 1, generator=gen, device=w.device)
    cdf = torch.special.ndtr(z / math.sqrt(3.0))
    index = torch.floor(levels * cdf - u).clamp(0, levels - 1)
    lower = torch.where(index == 0, torch.zeros_like(u), (index + u) / levels)
    upper = torch.where(index == levels - 1, torch.ones_like(u), (index + 1 + u) / levels)
    mid = (0.5 * (lower + upper)).clamp(1e-7, 1 - 1e-7)
    coded = math.sqrt(3.0) * torch.special.ndtri(mid) / math.sqrt(group)
    recon = norm * inverse_random_rotation(coded, seed)
    return recon.reshape(out_f, -1)[:, :in_f].to(weight.dtype)
