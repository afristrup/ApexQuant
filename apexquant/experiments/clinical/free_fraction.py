"""The free share of GPU memory, over every visible device."""

from __future__ import annotations

import torch


def free_fraction() -> float:
    """The smallest ``free / total`` over the visible GPUs."""
    return min(free / total for free, total in
               (torch.cuda.mem_get_info(d) for d in range(torch.cuda.device_count())))
