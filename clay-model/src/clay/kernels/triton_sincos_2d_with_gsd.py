"""Launch the GSD-scaled 2-D sincos position-table kernel."""

from __future__ import annotations

import torch
import triton

from ..ops.gsd_frequency_ladder import gsd_frequency_ladder
from .sincos_2d_kernel import sincos_2d_kernel

SINCOS_QUARTERS = 4


def triton_sincos_2d_with_gsd(  # noqa: PLR0913
    h: int, w: int, dim: int, gsd: torch.Tensor, temperature: int, device: torch.device
) -> torch.Tensor:
    """Build the ``(h * w, dim)`` position table on the device.

    Args:
        h: Patch-grid height.
        w: Patch-grid width.
        dim: Encoding width, a multiple of four.
        gsd: Scalar ground sample distance, which stretches the frequency ladder.
        temperature: Base of the geometric frequency ladder.
        device: CUDA device to build the table on.

    Returns:
        Tensor of shape ``(h * w, dim)``.
    """
    if dim % SINCOS_QUARTERS != 0:
        raise ValueError(f"dim {dim} must be a multiple of {SINCOS_QUARTERS}")

    quarter = dim // SINCOS_QUARTERS
    omega = gsd_frequency_ladder(quarter, dim, gsd, temperature, device)
    out = torch.empty(h * w, dim, device=device, dtype=torch.float32)
    sincos_2d_kernel[(h * w,)](
        out,
        omega,
        w,
        quarter,
        out.stride(0),
        BLOCK_Q=triton.next_power_of_2(quarter),
    )
    return out
