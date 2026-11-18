"""Graph-safe runtime dispatch for packed multi-pass E8P linears."""

from __future__ import annotations

import functools

import torch

from ..kernels.triton_gemv_e8p_real import triton_gemv_e8p_real
from ..vq.e8p_codec import E8PCodec
from .float32_default import float32_default


@functools.lru_cache(maxsize=None)
def e8p_codec(group_size: int, device: str) -> E8PCodec:
    """The codec every E8P layer of this group size encodes with, built once."""
    with float32_default():
        return E8PCodec(group_size, seed=0, device=device)


@torch.library.custom_op("apexquant::e8p_linear", mutates_args=(), device_types="cuda")
def e8p_linear(rotated: torch.Tensor, words: torch.Tensor, scale: torch.Tensor,
               group_size: int) -> torch.Tensor:
    """``sum_p rotated[p] @ C_p^T`` for E8P passes ``C_p`` stored as packed words.

    ``rotated`` is ``(P, T, K)``: the activations under each pass's rotation,
    so no inverse rotation is needed. One token uses the real-E8P GEMV per pass;
    more tokens decode each pass to a transient FP16 matrix and multiply.
    """
    codec = e8p_codec(group_size, str(rotated.device))
    passes, tokens, k = rotated.shape
    n = words.shape[1]
    if tokens == 1:
        out = torch.zeros(n, device=rotated.device, dtype=torch.float32)
        for p in range(passes):
            out += triton_gemv_e8p_real(rotated[p, 0].contiguous(), words[p], scale[p],
                                        codec._S_f32, codec._par_i32, group_size,
                                        codec_scale=codec.scale).view(n)
        return out.view(1, n)
    out = torch.zeros(tokens, n, device=rotated.device, dtype=torch.float32)
    for p in range(passes):
        codes = torch.stack([words[p] & 0xFFFF, (words[p] >> 16) & 0xFFFF], dim=-1)
        values = codec.decode(codes.reshape(n, -1)).view(n, -1, group_size)
        dense = (values * scale[p].unsqueeze(-1)).view(n, k).half()
        out += (rotated[p].half() @ dense.T).float()
    return out


@e8p_linear.register_fake
def _e8p_linear_fake(rotated: torch.Tensor, words: torch.Tensor, scale: torch.Tensor,
                     group_size: int) -> torch.Tensor:
    return rotated.new_empty((rotated.shape[1], words.shape[1]), dtype=torch.float32)
