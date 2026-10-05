"""Mid-M fused GEMM: all passes, split-K, group-aligned tiles (Clay's regime).

The original Triton GEMM is 3.5--5x off cuBLAS at Clay's 65-token forward. Four
structural reasons, each fixed here:

* **Occupancy.** ``(M=65, N=1024)`` gives ``2 x 16 = 32`` programs on 128 SMs.
  Here the pass index and a group-aligned K-split join the grid (partial sums
  via fp32 atomics), multiplying the program count until the device fills.
* **Per-element scale loads.** The old kernel resolved ``k // GROUP_SIZE`` per
  element (``BLOCK_N x BLOCK_K`` scale loads per tile). ``BLOCK_K`` divides the
  group size and chunks are group-aligned, so a K-tile lies in exactly ONE
  group: the scale is one ``(BLOCK_N,)`` vector load per tile.
* **M-tile waste.** ``BLOCK_M = 32`` fits 65 tokens in 3 tiles instead of 2
  half-empty 64-row tiles, and yields more programs.
* **Launch overhead.** P passes ran as P kernel launches with P output
  round-trips; here one launch covers every pass.

Same numerics as the reference (ieee fp32 ``tl.dot``); results match the torch
path to float accumulation order.
"""

from __future__ import annotations

import functools

import torch

try:
    import triton
    import triton.language as tl

    GEMM_MID_AVAILABLE = True
except ImportError:  # pragma: no cover
    GEMM_MID_AVAILABLE = False


if GEMM_MID_AVAILABLE:

    @triton.jit
    def _gemm_mid_kernel(
        rx_ptr, words_ptr, scale_ptr, levels_ptr, out_ptr,
        M, N, K, words_per_row, words_per_chunk,
        stride_rp, stride_rm,             # rx (P, M, K), K contiguous
        stride_wp, stride_wn,             # words (P, N, W)
        stride_sp, stride_sn, stride_sg,  # scale (P, N, G)
        stride_om, stride_on,
        GROUP_SIZE: tl.constexpr, BITS: tl.constexpr, VPW: tl.constexpr,
        SPLIT_K: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_z = tl.program_id(2)
        p = pid_z // SPLIT_K
        chunk = pid_z % SPLIT_K

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        n_mask = offs_n < N

        k_lo = chunk * words_per_chunk * VPW
        k_hi = tl.minimum(k_lo + words_per_chunk * VPW, K)

        rx_base = rx_ptr + p * stride_rp + offs_m[:, None] * stride_rm
        wrd_base = words_ptr + p * stride_wp + offs_n[:, None] * stride_wn
        sc_base = scale_ptr + p * stride_sp + offs_n * stride_sn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(k_lo, k_hi, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            # Chunks are group-aligned and BLOCK_K divides both GROUP_SIZE and
            # the chunk length, so the tile is full and lies in one group.
            rx = tl.load(rx_base + offs_k[None, :],
                         mask=m_mask[:, None], other=0.0)
            word = tl.load(wrd_base + (offs_k[None, :] // VPW),
                           mask=n_mask[:, None], other=0)
            idx = (word >> (BITS * (offs_k[None, :] % VPW))) & ((1 << BITS) - 1)
            code = tl.load(levels_ptr + idx)
            sc = tl.load(sc_base + (k0 // GROUP_SIZE) * stride_sg,
                         mask=n_mask, other=0.0)          # one group per tile
            acc += tl.dot(rx, tl.trans(code * sc[:, None]))

        out = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.atomic_add(out, acc, mask=m_mask[:, None] & n_mask[None, :])


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def triton_gemm_mid(rx: torch.Tensor, words: torch.Tensor, scale: torch.Tensor,
                    levels: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    """All-passes fused GEMM: ``out[m,n] = sum_p sum_k rx[p,m,k] * w_p[n,k]``.

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
    if not GEMM_MID_AVAILABLE:
        raise RuntimeError("triton is not available")
    P, M, K = rx.shape
    N = words.shape[1]
    words_per_row = words.shape[2]
    vpw = 32 // bits
    wpg = group_size // vpw

    # Swept on RTX 4090 (Clay shapes, M=65): BK=64 with num_stages=2 is the
    # optimum -- deeper pipelining (stages>=3) spills registers on the int32
    # word tiles and costs 3x. BM drops to 32 when a single tile covers M.
    block_m = 32 if M <= 32 else 64
    block_n, block_k = 64, 64
    tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    sms = _sm_count(rx.device.index or 0)
    max_chunks = max(1, words_per_row // wpg)
    split_k = min(max(1, (2 * sms) // max(1, tiles * P)), max_chunks)
    words_per_chunk = triton.cdiv(triton.cdiv(words_per_row, split_k), wpg) * wpg

    out = torch.zeros((M, N), dtype=torch.float32, device=rx.device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n), split_k * P)
    _gemm_mid_kernel[grid](
        rx, words, scale, levels, out,
        M, N, K, words_per_row, words_per_chunk,
        rx.stride(0), rx.stride(1),
        words.stride(0), words.stride(1),
        scale.stride(0), scale.stride(1), scale.stride(2),
        out.stride(0), out.stride(1),
        GROUP_SIZE=group_size, BITS=bits, VPW=vpw, SPLIT_K=split_k,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=4, num_stages=2,
    )
    return out
