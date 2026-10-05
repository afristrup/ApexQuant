"""Multi-pass split-K fused GEMV -- one launch for ALL residual passes.

Why the single-pass GEMV loses beyond one pass, measured on real 7B shapes:

* **Launch and accumulate overhead scales with P.** The per-pass path does P
  rotations, P kernel launches and P output read-modify-writes per layer; at
  ~5 us a launch that is milliseconds per token on a 224-linear decoder stack.
  Here the pass dimension is folded into the grid, so a P-pass layer is ONE
  launch against a stacked ``rx`` of every pass's rotated activation.

* **Down-projections under-fill the GPU.** With ``N = 4096`` output rows and
  ``BLOCK_N = 64``, a row-parallel GEMV makes 64 programs for 128 SMs, and the
  huge fan-in (K = 14336-18944) is walked serially by each. Splitting K across
  ``SPLIT_K`` chunks (aligned to scale groups) multiplies the program count;
  partial sums combine with one fp32 ``atomic_add`` per row block.

The grid is ``(cdiv(N, BLOCK_N), SPLIT_K * P)``: each program dequantises and
reduces one (rows x K-chunk) tile of one pass. Everything else -- int32 word
loads (16 two-bit indices per word), one scale read per word, codebook gather --
matches the single-pass kernel.
"""

from __future__ import annotations

import functools

import torch

try:
    import triton
    import triton.language as tl

    GEMV_MP_AVAILABLE = True
except ImportError:  # pragma: no cover
    GEMV_MP_AVAILABLE = False


if GEMV_MP_AVAILABLE:

    @triton.jit
    def _gemv_mp_kernel(
        rx_ptr, words_ptr, scale_ptr, levels_ptr, out_ptr,
        N, K, words_per_row, words_per_chunk,
        stride_rp,                       # rx (P, K): pass stride
        stride_wp, stride_wn,            # words (P, N, W)
        stride_sp, stride_sn, stride_sg,  # scale (P, N, G)
        GROUP_SIZE: tl.constexpr, BITS: tl.constexpr, VPW: tl.constexpr,
        SPLIT_K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_z = tl.program_id(1)
        p = pid_z // SPLIT_K             # residual pass
        chunk = pid_z % SPLIT_K          # K-chunk within the pass
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        w_lo = chunk * words_per_chunk
        w_hi = tl.minimum(w_lo + words_per_chunk, words_per_row)

        rx_base = rx_ptr + p * stride_rp
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
            g = (offs_w * VPW) // GROUP_SIZE  # a word lies in one scale group
            sc = tl.load(sc_base + g[None, :] * stride_sg,
                         mask=n_mask[:, None] & w_mask[None, :], other=0.0)
            acc += tl.sum(wsum * sc, axis=1)
        tl.atomic_add(out_ptr + offs_n, acc, mask=n_mask)


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    # get_device_properties is a cudart call (~tens of us); never pay it per GEMV.
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def triton_gemv_mp(rx: torch.Tensor, words: torch.Tensor, scale: torch.Tensor,
                   levels: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    """All-passes fused mat-vec: ``out[n] = sum_p sum_k rx[p,k] * w_p[n,k]``.

    Args:
        rx: Rotated activations, one row per pass ``(P, K)`` float32.
        words: Packed indices viewed as int32 words ``(P, N, K*bits/32)``.
        scale: Per-group scales ``(P, N, K/group_size)`` float32.
        levels: Codebook ``(2**bits,)`` float32.
        group_size: Scale group size (multiple of ``32 // bits``).
        bits: Index bit-width.

    Returns:
        Output ``(1, N)`` float32.
    """
    if not GEMV_MP_AVAILABLE:
        raise RuntimeError("triton is not available")
    P, K = rx.shape
    N = words.shape[1]
    words_per_row = words.shape[2]
    vpw = 32 // bits
    wpg = group_size // vpw  # words per scale group

    block_n, block_w = 64, 32
    n_blocks = triton.cdiv(N, block_n)
    # Fill the device: aim for ~2 programs per SM across passes and chunks,
    # keeping chunk boundaries aligned to scale groups.
    sms = _sm_count(rx.device.index or 0)
    max_chunks = max(1, words_per_row // wpg)
    split_k = min(max(1, (2 * sms) // max(1, n_blocks * P)), max_chunks)
    words_per_chunk = triton.cdiv(triton.cdiv(words_per_row, split_k), wpg) * wpg

    out = torch.zeros((1, N), dtype=torch.float32, device=rx.device)
    grid = (n_blocks, split_k * P)
    _gemv_mp_kernel[grid](
        rx, words, scale, levels, out,
        N, K, words_per_row, words_per_chunk,
        rx.stride(0),
        words.stride(0), words.stride(1),
        scale.stride(0), scale.stride(1), scale.stride(2),
        GROUP_SIZE=group_size, BITS=bits, VPW=vpw,
        SPLIT_K=split_k, BLOCK_N=block_n, BLOCK_W=block_w,
        num_warps=4, num_stages=3,
    )
    return out
