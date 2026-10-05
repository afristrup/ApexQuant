"""Fused GEMV against a *generated* E8-style lattice code, with a resident table.

`triton_gemv_e8` measures the decode cost of the obvious thing: store all 2^16
lattice points and gather. That costs 1.6-3.8x the scalar decode, and the reason
is structural rather than algorithmic -- 2^16 x 8 fp16 is 1 MiB against 48 KiB of
shared memory per block, so every coordinate is an L2 gather.

Reporting that number alone would be measuring a strawman, because it is exactly
the cost that QuIP#'s E8P is designed to avoid: E8P *generates* its codewords
from a small source codebook rather than storing them, and QTIP goes further and
computes codewords with no table at all. This kernel measures the other end of
that bracket. A 16-bit code is split

    idx   = code & 0xFF        8 bits, an entry of a 256-word source table
    signs = code >> 8          8 bits, one sign flip per coordinate

so the same 16 bits still name one of 2^16 vectors in 8 dimensions, at the same
two bits per coordinate and the same weight bandwidth, but the table is
256 x 8 fp16 = 4 KiB and stays resident in L1. The decode pays a little integer
arithmetic (a shift, a mask, a negate) in exchange for not leaving the SM.

Scope. This measures *decode cost* at a fixed rate, and is deliberately not a
claim to reproduce QuIP#'s E8P construction, whose source codebook and padding
are chosen for the lattice's packing properties rather than for this benchmark;
the distortion of the finite E8 codebook is measured separately in
experiments/vector_stage_rho.py. What the pair of kernels brackets is the range a
lattice stage's decode can occupy, from a naive stored table to a generated one.
"""

from __future__ import annotations

import functools

import torch

try:
    import triton
    import triton.language as tl

    GEMV_E8P_AVAILABLE = True
except ImportError:  # pragma: no cover
    GEMV_E8P_AVAILABLE = False


if GEMV_E8P_AVAILABLE:

    @triton.jit
    def _gemv_e8p_kernel(
        rx_ptr, words_ptr, scale_ptr, src_ptr, out_ptr,
        N, K, words_per_row, words_per_chunk,
        stride_wn, stride_sn, stride_sg,
        GROUP_SIZE: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr,
        D: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        chunk = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        w_lo = chunk * words_per_chunk
        w_hi = tl.minimum(w_lo + words_per_chunk, words_per_row)

        wrd_base = words_ptr + offs_n[:, None] * stride_wn
        sc_base = scale_ptr + offs_n[:, None] * stride_sn
        offs_d = tl.arange(0, D)

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for w0 in range(w_lo, w_hi, BLOCK_W):
            offs_w = w0 + tl.arange(0, BLOCK_W)
            w_mask = offs_w < w_hi
            word = tl.load(wrd_base + offs_w[None, :],
                           mask=n_mask[:, None] & w_mask[None, :], other=0)
            wsum = tl.zeros((BLOCK_N, BLOCK_W), dtype=tl.float32)
            # two 16-bit codes per word; each generates one D-dim vector
            for h in tl.static_range(2):
                code = (word >> (16 * h)) & 0xFFFF              # (BN, BW)
                idx = code & 0xFF                               # source entry
                sgn = (code >> 8) & 0xFF                        # sign bits
                # 4 KiB source table: this gather stays in L1
                sp = src_ptr + idx[:, :, None] * D + offs_d[None, None, :]
                base = tl.load(sp).to(tl.float32)               # (BN, BW, D)
                bit = (sgn[:, :, None] >> offs_d[None, None, :]) & 1
                vec = base * (1.0 - 2.0 * bit.to(tl.float32))
                k0 = (offs_w * 2 + h) * D
                kk = k0[:, None] + offs_d[None, :]
                rxv = tl.load(rx_ptr + kk, mask=kk < K, other=0.0)
                wsum += tl.sum(vec * rxv[None, :, :], axis=2)
            g = (offs_w * 2 * D) // GROUP_SIZE
            sc = tl.load(sc_base + g[None, :] * stride_sg,
                         mask=n_mask[:, None] & w_mask[None, :], other=0.0)
            acc += tl.sum(wsum * sc, axis=1)
        tl.atomic_add(out_ptr + offs_n, acc, mask=n_mask)


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def triton_gemv_e8p(rx: torch.Tensor, words: torch.Tensor, scale: torch.Tensor,
                    source: torch.Tensor, group_size: int) -> torch.Tensor:
    """Fused mat-vec against a generated (sign-flip) lattice code.

    Args:
        rx: Rotated activations ``(K,)`` float32.
        words: Packed 16-bit codes as int32 words ``(N, K/16)``, two per word,
            identical bandwidth to the 2-bit scalar and stored-table E8 paths.
        scale: Per-group scales ``(N, K/group_size)`` float32.
        source: Source codebook ``(256, 8)``; 4 KiB in fp16, L1-resident.
        group_size: Scale group size.

    Returns:
        Output ``(1, N)`` float32.
    """
    if not GEMV_E8P_AVAILABLE:
        raise RuntimeError("triton is not available")
    K = rx.shape[-1]
    N, words_per_row = words.shape
    D = source.shape[1]
    vals_per_word = 2 * D
    wpg = group_size // vals_per_word

    block_n, block_w = 64, 8
    n_blocks = triton.cdiv(N, block_n)
    sms = _sm_count(rx.device.index or 0)
    max_chunks = max(1, words_per_row // max(1, wpg))
    split_k = min(max(1, (2 * sms) // max(1, n_blocks)), max_chunks)
    words_per_chunk = triton.cdiv(triton.cdiv(words_per_row, split_k),
                                  max(1, wpg)) * max(1, wpg)

    out = torch.zeros((1, N), dtype=torch.float32, device=rx.device)
    _gemv_e8p_kernel[(n_blocks, split_k)](
        rx, words, scale, source, out,
        N, K, words_per_row, words_per_chunk,
        words.stride(0), scale.stride(0), scale.stride(1),
        GROUP_SIZE=group_size, BLOCK_N=block_n, BLOCK_W=block_w, D=D,
        num_warps=4, num_stages=3,
    )
    return out
