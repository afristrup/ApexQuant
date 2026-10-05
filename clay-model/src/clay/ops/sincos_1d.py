"""Sinusoidal encoding of a 1-D quantity, used for band wavelengths."""

from __future__ import annotations

import torch

from ..constants import SINCOS_TEMPERATURE

SINCOS_HALVES = 2


def sincos_1d(
    values: torch.Tensor, dim: int, temperature: int = SINCOS_TEMPERATURE
) -> torch.Tensor:
    """Encode each entry of ``values`` as ``dim`` interleaved sin/cos features.

    Clay calls this on the band wavelengths of whichever sensor produced the
    chip, which is what lets one set of weights read Sentinel-2, Landsat, SAR and
    NAIP. It stays on the torch path: the table is one row per band, roughly ten
    rows wide, so a kernel launch would cost more than the arithmetic.

    Args:
        values: Tensor of shape ``(N,)``.
        dim: Output width, a multiple of two.
        temperature: Base of the geometric frequency ladder.

    Returns:
        Tensor of shape ``(N, dim)``.
    """
    if dim % SINCOS_HALVES != 0:
        raise ValueError(f"dim {dim} must be a multiple of {SINCOS_HALVES}")

    half = dim // SINCOS_HALVES
    ladder = torch.arange(half, device=values.device) / (half - 1)
    omega = 1.0 / (temperature**ladder)
    angles = values[:, None] * omega[None, :]
    return torch.cat((angles.sin(), angles.cos()), dim=1).to(torch.float32)
