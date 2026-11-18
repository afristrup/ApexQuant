"""Graph-safe op: Sinkhorn column scale fused into the trellis input rotation."""

from __future__ import annotations

import torch

from ..rotation.triton_wht_rotate_scaled import triton_wht_rotate_scaled


@torch.library.custom_op("apexquant::trellis_rotate_scaled", mutates_args=(),
                         device_types="cuda")
def trellis_rotate_scaled(x: torch.Tensor, col: torch.Tensor, signs: torch.Tensor,
                          group_size: int) -> torch.Tensor:
    """``(passes, tokens, width)`` rotation of ``x * col`` in float32."""
    rotated = triton_wht_rotate_scaled(x.float().contiguous(), col, group_size, signs)
    return rotated.view(signs.shape[0], x.shape[0], -1)


@trellis_rotate_scaled.register_fake
def _trellis_rotate_scaled_fake(x: torch.Tensor, col: torch.Tensor, signs: torch.Tensor,
                                group_size: int) -> torch.Tensor:
    return x.new_empty((signs.shape[0], x.shape[0], x.shape[1]), dtype=torch.float32)
