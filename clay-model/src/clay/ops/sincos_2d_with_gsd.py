"""GSD-scaled 2-D sinusoidal position table for a square patch grid."""

from __future__ import annotations

import torch

from ..constants import SINCOS_TEMPERATURE
from .gsd_frequency_ladder import gsd_frequency_ladder
from .use_triton import use_triton

SINCOS_QUARTERS = 4


def sincos_2d_with_gsd(  # noqa: PLR0913
    h: int,
    w: int,
    dim: int,
    gsd: torch.Tensor,
    device: torch.device,
    temperature: int = SINCOS_TEMPERATURE,
) -> torch.Tensor:
    """Build the position encoding for an ``h x w`` patch grid.

    Scaling the frequency ladder by the ground sample distance is what makes the
    encoding physical rather than merely positional: a 10 m Sentinel-2 patch and
    a 0.3 m NAIP patch at the same grid index land at different phases, so the
    model reads distance on the ground instead of distance in patches.

    Args:
        h: Patch-grid height.
        w: Patch-grid width.
        dim: Output width, a multiple of four.
        gsd: Scalar ground sample distance in metres.
        device: Device the table is built on.
        temperature: Base of the geometric frequency ladder.

    Returns:
        Tensor of shape ``(h * w, dim)``.
    """
    if dim % SINCOS_QUARTERS != 0:
        raise ValueError(f"dim {dim} must be a multiple of {SINCOS_QUARTERS}")

    gsd = gsd.to(device)
    if use_triton(gsd):
        from ..kernels.triton_sincos_2d_with_gsd import triton_sincos_2d_with_gsd

        return triton_sincos_2d_with_gsd(h, w, dim, gsd, temperature, device)

    quarter = dim // SINCOS_QUARTERS
    rows, cols = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    omega = gsd_frequency_ladder(quarter, dim, gsd, temperature, device)

    col_angle = cols.flatten()[:, None] * omega[None, :]
    row_angle = rows.flatten()[:, None] * omega[None, :]
    table = torch.cat(
        (col_angle.sin(), col_angle.cos(), row_angle.sin(), row_angle.cos()), dim=1
    )
    return table.to(torch.float32)
