"""The geometric frequency ladder of the 2-D position encoding, scaled by GSD."""

from __future__ import annotations

import torch


def gsd_frequency_ladder(
    quarter: int, dim: int, gsd: torch.Tensor, temperature: int, device: torch.device
) -> torch.Tensor:
    """Return the ``(quarter,)`` angular frequencies of the position encoding.

    Both the torch path and the Triton kernel take the ladder from here rather
    than each deriving it. That is not only tidiness: recomputing it as
    ``exp(-log(t) * 2c / dim)`` inside a kernel drifts from torch's ``t ** x`` by
    a few units in the last place, and since the coarsest frequency multiplies a
    patch index the drift lands in the argument of a sine, where it is amplified
    into a visible error.

    Args:
        quarter: Number of frequencies, one quarter of the encoding width.
        dim: Full encoding width.
        gsd: Scalar ground sample distance in metres.
        temperature: Base of the ladder.
        device: Device to build on.

    Returns:
        Tensor of shape ``(quarter,)``.
    """
    ladder = torch.arange(quarter, device=device) / (quarter - 1)
    return 1.0 / (temperature ** (2 * ladder / dim)) * gsd.to(device)
