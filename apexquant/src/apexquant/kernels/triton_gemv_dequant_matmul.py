"""Fused dequant + mat-vec for batch-1 decode -- the regime where 2-bit beats bf16.

The GEMM kernel tiles the output with ``BLOCK_M=64`` and a tensor-core ``tl.dot``.
At ``M=1`` that pads M to 64 and does ~64x wasted work, so it runs compute-bound
at ~1/24th of the memory roofline and loses to a dense bf16 GEMV. This is a true
mat-vec: no ``tl.dot``, one program per block of output rows reducing against the
single activation vector. Three data-movement ideas make it ~4-6x dense bf16 on a
layer whose bf16 weight spills L2:

* load the packed weight as **int32 words** (32/bits indices/word) -- each byte read
  once;
* a word lies in one scale group (``group_size`` is a multiple of the values/word),
  so its scale is read **once per word**;
* a large ``BLOCK_N`` **amortizes the activation reads**: the mat-vec re-reads ``rx``
  once per row-block, so ``rx`` traffic is ``(N/BLOCK_N)*K`` -- the dominant cost at
  small BLOCK_N. Looping over the values-in-a-word (2-D tiles, no 3-D intermediate)
  keeps register pressure low enough for BLOCK_N=64.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    GEMV_TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    GEMV_TRITON_AVAILABLE = False


if GEMV_TRITON_AVAILABLE:

    @triton.jit
    def _gemv_kernel(
        rx_ptr, words_ptr, scale_ptr, levels_ptr, out_ptr,
        N, K, words_per_row,
        stride_wn, stride_wk, stride_sn, stride_sg,
        GROUP_SIZE: tl.constexpr, BITS: tl.constexpr, VALS_PER_WORD: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for w0 in range(0, words_per_row, BLOCK_W):
            offs_w = w0 + tl.arange(0, BLOCK_W)
            w_mask = offs_w < words_per_row
            word = tl.load(
                words_ptr + offs_n[:, None] * stride_wn + offs_w[None, :] * stride_wk,
                mask=(offs_n[:, None] < N) & w_mask[None, :], other=0,
            )  # (BLOCK_N, BLOCK_W) int32 -- each byte read once
            wsum = tl.zeros((BLOCK_N, BLOCK_W), dtype=tl.float32)
            for p in tl.static_range(VALS_PER_WORD):
                idx = (word >> (BITS * p)) & ((1 << BITS) - 1)
                code = tl.load(levels_ptr + idx)                 # (BLOCK_N, BLOCK_W)
                k = offs_w * VALS_PER_WORD + p
                rxv = tl.load(rx_ptr + k, mask=k < K, other=0.0)  # (BLOCK_W,)
                wsum += code * rxv[None, :]
            # A word is inside one scale group, so scale is one value per word.
            g = (offs_w * VALS_PER_WORD) // GROUP_SIZE
            sc = tl.load(
                scale_ptr + offs_n[:, None] * stride_sn + g[None, :] * stride_sg,
                mask=(offs_n[:, None] < N) & w_mask[None, :], other=0.0,
            )
            acc += tl.sum(wsum * sc, axis=1)
        tl.store(out_ptr + offs_n, acc, mask=offs_n < N)


def triton_gemv_dequant_matmul(rx: torch.Tensor, packed: torch.Tensor,
                               scale: torch.Tensor, levels: torch.Tensor,
                               group_size: int, bits: int = 4) -> torch.Tensor:
    """Single-row fused dequant mat-vec: ``out[0,n] = sum_k rx[0,k] * w[n,k]``.

    Same semantics and reconstruction as :func:`triton_fused_dequant_matmul`, but
    memory-bound for ``M == 1``. Requires ``rx`` to have a single row, the packed
    row length to be a multiple of 4 bytes, and ``group_size`` a multiple of
    ``32 // bits`` -- both always hold for the dispatched ``group_size >= 16``.
    """
    if not GEMV_TRITON_AVAILABLE:
        raise RuntimeError("triton is not available")

    _M, K = rx.shape
    N = packed.shape[0]
    words = packed.view(torch.int32)  # (N, K * bits / 32); each word packs 32/bits indices
    words_per_row = words.shape[1]
    vals_per_word = 32 // bits
    out = torch.empty((1, N), dtype=torch.float32, device=rx.device)
    # BLOCK_N=64 amortizes the rx reads (the dominant traffic), BLOCK_W=32, 4 warps.
    # Larger BLOCK_N / autotuning looked better in an isolated microbenchmark but
    # regressed in the real forward path (rotation + cache state) -- fixed 64 is the
    # robust optimum: ~3.0x (d=8192) to ~5.1x (d=16384) dense bf16.
    block_n, block_w = 64, 32
    grid = (triton.cdiv(N, block_n),)
    _gemv_kernel[grid](
        rx, words, scale, levels, out,
        N, K, words_per_row,
        words.stride(0), words.stride(1), scale.stride(0), scale.stride(1),
        GROUP_SIZE=group_size, BITS=bits, VALS_PER_WORD=vals_per_word,
        BLOCK_N=block_n, BLOCK_W=block_w,
        num_warps=4, num_stages=3,
    )
    return out
