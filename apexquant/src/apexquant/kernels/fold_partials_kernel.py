"""Triton kernel: sum a tile across bands and apply the epilogue."""

from __future__ import annotations

import triton
import triton.language as tl

# 1 / sqrt(2), the argument scale of the exact (erf) GELU.
FOLD_INV_SQRT2 = tl.constexpr(0.7071067811865476)


@triton.jit
def fold_partials_kernel(  # noqa: PLR0913
    parts_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    bands,
    stride_band,
    HAS_BIAS: tl.constexpr,
    ACT_GELU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reduce ``bands`` partial results into one tile, then bias and activate.

    Split-K and multi-pass launches leave one band per (chunk, pass), and folding
    them is a separate pass over the output whatever happens. Doing the bias and
    the activation here rather than afterwards makes it the *only* extra pass:
    the alternative is a sum, then an add, then an activation, three reads and
    three writes of a tile that is already in registers once.

    Bands are summed in index order, so two identical calls return identical bits
    -- the same reason the GEMM writes bands instead of using ``atomic_add``.
    """
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    keep = (offs_m < M)[:, None] & (offs_n < N)[None, :]

    tile = offs_m.to(tl.int64)[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for band in range(bands):
        acc += tl.load(
            parts_ptr + band.to(tl.int64) * stride_band + tile,
            mask=keep, other=0.0,
        )

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
    if ACT_GELU:
        acc = 0.5 * acc * (1.0 + tl.math.erf(acc * FOLD_INV_SQRT2))

    tl.store(out_ptr + tile, acc, mask=keep)
