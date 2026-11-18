"""Apply a group-wise quantizer along whichever side of a matrix the group divides."""

from __future__ import annotations

from collections.abc import Callable

import torch


def grouped_axis(w: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor], group: int) -> torch.Tensor:
    """``fn(w)`` when ``group`` divides the fan-in, else ``fn(w.T).T`` when it divides the fan-out.

    Grouping the output side rotates outputs instead of inputs, which a fused
    kernel serves equally well, and keeps every pass at exactly ``2 + 16 / group``
    bits instead of padding. A matrix divisible on neither side fails.
    """
    out_f, in_f = w.shape
    if in_f % group == 0:
        return fn(w)
    if out_f % group == 0:
        return fn(w.T.contiguous()).T.contiguous()
    raise ValueError(f"neither side of a {out_f}x{in_f} matrix divides group {group}")
