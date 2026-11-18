"""Fused online encoder for compact E8P uint16 codes."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _encode_kernel(x_ptr, source_ptr, parity_ptr, out_ptr, n, scale,
                       BLOCK: tl.constexpr):
        row = tl.program_id(0)
        d = tl.arange(0, 8)
        x = tl.load(x_ptr + row * 8 + d).to(tl.float32) / scale
        best = tl.full((), float("inf"), tl.float32)
        best_j = tl.zeros((), tl.int32)
        best_c = tl.zeros((), tl.int32)
        for j in range(256):
            s = tl.load(source_ptr + j * 8 + d)
            a2 = tl.zeros((), tl.float32)
            p2 = tl.full((), float("inf"), tl.float32)
            for cbit in range(2):
                shift = 0.25 if cbit == 0 else -0.25
                y = x - shift
                a = tl.abs(y)
                dist = tl.sum((a - s) * (a - s), axis=0)
                neg = tl.sum((y < 0).to(tl.int32), axis=0) & 1
                parity = tl.load(parity_ptr + j).to(tl.int32)
                penalty = tl.min(4.0 * a * s, axis=0)
                cost = dist + tl.where(neg != parity, penalty, 0.0)
                take = cost < best
                best = tl.where(take, cost, best)
                best_j = tl.where(take, j, best_j)
                best_c = tl.where(take, cbit, best_c)
        s = tl.load(source_ptr + best_j * 8 + d)
        shift = tl.where(best_c == 0, 0.25, -0.25)
        y = x - shift
        a = tl.abs(y)
        neg = tl.sum((y < 0).to(tl.int32), axis=0) & 1
        parity = tl.load(parity_ptr + best_j).to(tl.int32)
        penalty = tl.min(4.0 * a * s, axis=0)
        mismatch = neg != parity
        flip_idx = tl.argmin(a * s, axis=0)
        flip = mismatch & (d == flip_idx)
        signs = (y < 0) ^ flip
        bits = tl.sum(tl.where(d < 7, signs.to(tl.int32) << d, 0), axis=0)
        code = best_j | (bits << 8) | (best_c << 15)
        tl.store(out_ptr + row, code.to(tl.uint16))


def triton_e8p_encode(x: torch.Tensor, source: torch.Tensor,
                      parity: torch.Tensor, scale: float) -> torch.Tensor:
    """Encode rows of eight coordinates into compact E8P codes."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    if x.ndim != 2 or x.shape[-1] != 8:
        raise ValueError(f"x must have shape (n, 8), got {tuple(x.shape)}")
    if source.shape != (256, 8) or parity.shape != (256,):
        raise ValueError("invalid E8P lookup tables")
    out = torch.empty(x.shape[0], dtype=torch.uint16, device=x.device)
    _encode_kernel[(x.shape[0],)](
        x.contiguous(), source.contiguous(), parity.contiguous(), out,
        x.shape[0], scale, BLOCK=8, num_warps=4,
    )
    return out
