"""Block GEMV: a few rows (2-16) at mat-vec cost -- the speculative verify shape.

Speculative verification is an M = k+1 forward. The tiled GEMM is hopeless
there (M=5 pads a 32-row tensor-core tile; measured 4.49x a single token on
the Mistral stack) and the mat-vec handles one row. This kernel puts the row
index in the launch grid of the multi-pass split-K mat-vec: each program
dequantises one (rows x K-chunk) strip of one pass FOR ONE ROW. The packed
words are then read once from HBM and served from L2 for the remaining rows --
at M = 5 every 3-pass Mistral layer's payload fits comfortably in an RTX
4090's 72 MB L2 (largest is 44 MB) -- so the block verify costs close to a
single decoded token instead of five.

Everything else (int32 word loads, one scale read per word, codebook gather,
group-aligned K-split, fp32 atomics) matches ``triton_gemv_mp``.
"""

from __future__ import annotations

import functools

import torch

try:
    import triton
    import triton.language as tl

    GEMV_BLOCK_AVAILABLE = True
except ImportError:  # pragma: no cover
    GEMV_BLOCK_AVAILABLE = False


if GEMV_BLOCK_AVAILABLE:

    @triton.jit
    def _gemv_block_kernel(
        rx_ptr, words_ptr, scale_ptr, levels_ptr, out_ptr,
        N, K, words_per_row, words_per_chunk,
        stride_rp, stride_rm,             # rx (P, M, K)
        stride_wp, stride_wn,             # words (P, N, W)
        stride_sp, stride_sn, stride_sg,  # scale (P, N, G)
        stride_om,                        # out (M, N)
        GROUP_SIZE: tl.constexpr, BITS: tl.constexpr, VPW: tl.constexpr,
        SPLIT_K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_z = tl.program_id(1)
        m = tl.program_id(2)              # one activation row per program
        p = pid_z // SPLIT_K
        chunk = pid_z % SPLIT_K

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        w_lo = chunk * words_per_chunk
        w_hi = tl.minimum(w_lo + words_per_chunk, words_per_row)

        rx_base = rx_ptr + p * stride_rp + m * stride_rm
        wrd_base = words_ptr + p * stride_wp + offs_n[:, None] * stride_wn
        sc_base = scale_ptr + p * stride_sp + offs_n[:, None] * stride_sn

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for w0 in range(w_lo, w_hi, BLOCK_W):
            offs_w = w0 + tl.arange(0, BLOCK_W)
            w_mask = offs_w < w_hi
            word = tl.load(wrd_base + offs_w[None, :],
                           mask=n_mask[:, None] & w_mask[None, :], other=0)
            wsum = tl.zeros((BLOCK_N, BLOCK_W), dtype=tl.float32)
            for v in tl.static_range(VPW):
                idx = (word >> (BITS * v)) & ((1 << BITS) - 1)
                code = tl.load(levels_ptr + idx)
                k = offs_w * VPW + v
                rxv = tl.load(rx_base + k, mask=k < K, other=0.0)
                wsum += code * rxv[None, :]
            g = (offs_w * VPW) // GROUP_SIZE
            sc = tl.load(sc_base + g[None, :] * stride_sg,
                         mask=n_mask[:, None] & w_mask[None, :], other=0.0)
            acc += tl.sum(wsum * sc, axis=1)
        tl.atomic_add(out_ptr + m * stride_om + offs_n, acc, mask=n_mask)


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def triton_gemv_block(rx: torch.Tensor, words: torch.Tensor, scale: torch.Tensor,
                      levels: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    """All-passes block mat-vec: ``out[m,n] = sum_p sum_k rx[p,m,k] * w_p[n,k]``.

    Args:
        rx: Rotated activations ``(P, M, K)`` float32, K contiguous.
        words: Packed indices as int32 words ``(P, N, K*bits/32)``.
        scale: Per-group scales ``(P, N, K/group_size)`` float32.
        levels: Codebook ``(2**bits,)`` float32.
        group_size: Scale group size.
        bits: Index bit-width.

    Returns:
        Output ``(M, N)`` float32.
    """
    if not GEMV_BLOCK_AVAILABLE:
        raise RuntimeError("triton is not available")
    P, M, K = rx.shape
    N = words.shape[1]
    words_per_row = words.shape[2]
    vpw = 32 // bits
    wpg = group_size // vpw

    block_n, block_w = 64, 32
    n_blocks = triton.cdiv(N, block_n)
    sms = _sm_count(rx.device.index or 0)
    max_chunks = max(1, words_per_row // wpg)
    split_k = min(max(1, (2 * sms) // max(1, n_blocks * P * M)), max_chunks)
    words_per_chunk = triton.cdiv(triton.cdiv(words_per_row, split_k), wpg) * wpg

    out = torch.zeros((M, N), dtype=torch.float32, device=rx.device)
    grid = (n_blocks, split_k * P, M)
    _gemv_block_kernel[grid](
        rx, words, scale, levels, out,
        N, K, words_per_row, words_per_chunk,
        rx.stride(0), rx.stride(1),
        words.stride(0), words.stride(1),
        scale.stride(0), scale.stride(1), scale.stride(2),
        out.stride(0),
        GROUP_SIZE=group_size, BITS=bits, VPW=vpw, SPLIT_K=split_k,
        BLOCK_N=block_n, BLOCK_W=block_w,
        num_warps=4, num_stages=2,
    )
    return out
