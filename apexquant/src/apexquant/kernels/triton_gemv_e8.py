"""Fused GEMV with an E8-lattice stage quantiser, for measuring its decode cost.

Section "The greedy tax is a property of the stage quantiser" reports that a
lattice stage cuts the per-pass distortion from 0.1171 to 0.0906 at the same two
bits per coordinate, and then concedes that we have not run one end to end
because "a trellis decode is a different engineering problem". This kernel makes
that concession quantitative for the lattice case, which is the cheaper of the
two to build.

The comparison is deliberately bandwidth-neutral. At two bits per coordinate:

  scalar path   16 two-bit indices per int32 word, codebook is 4 fp32 values and
                lives in registers; one gather per coordinate hits nothing.
  E8 path       two 16-bit codes per int32 word, each code naming one of 2^16
                lattice points in 8 dimensions, so a word still covers 16
                coordinates. Same words per row, same scale reads, same HBM
                traffic for the weights.

Everything else is held fixed, so any latency difference is decode cost alone.

The cost is structural rather than incidental, which is the point worth
measuring: the codebook is 2^16 x 8 fp16 = 1 MiB. Shared memory on Ada is about
100 KiB per SM, so unlike the 4-entry scalar codebook this table cannot be made
resident; every decoded coordinate is an L2 gather. That is precisely why QuIP#
gives E8P a structured construction it can generate rather than store, and why
QTIP moves to *computed* codes with no table at all. This kernel measures what
the naive table-resident-in-global version costs, which lower-bounds how much
work a deployable lattice stage would have to avoid.
"""

from __future__ import annotations

import functools

import torch

try:
    import triton
    import triton.language as tl

    GEMV_E8_AVAILABLE = True
except ImportError:  # pragma: no cover
    GEMV_E8_AVAILABLE = False


if GEMV_E8_AVAILABLE:

    @triton.jit
    def _gemv_e8_kernel(
        rx_ptr, words_ptr, scale_ptr, cb_ptr, out_ptr,
        N, K, words_per_row, words_per_chunk,
        stride_wn, stride_sn, stride_sg,
        GROUP_SIZE: tl.constexpr, SPLIT_K: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr, D: tl.constexpr,
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
            # two 16-bit lattice codes per word, each naming one D-dim point
            for h in tl.static_range(2):
                idx = (word >> (16 * h)) & 0xFFFF               # (BN, BW)
                cbp = cb_ptr + idx[:, :, None] * D + offs_d[None, None, :]
                code = tl.load(cbp).to(tl.float32)              # (BN, BW, D)
                k0 = (offs_w * 2 + h) * D                       # (BW,)
                kk = k0[:, None] + offs_d[None, :]              # (BW, D)
                rxv = tl.load(rx_ptr + kk, mask=kk < K, other=0.0)
                wsum += tl.sum(code * rxv[None, :, :], axis=2)
            g = (offs_w * 2 * D) // GROUP_SIZE
            sc = tl.load(sc_base + g[None, :] * stride_sg,
                         mask=n_mask[:, None] & w_mask[None, :], other=0.0)
            acc += tl.sum(wsum * sc, axis=1)
        tl.atomic_add(out_ptr + offs_n, acc, mask=n_mask)


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def triton_gemv_e8(rx: torch.Tensor, words: torch.Tensor, scale: torch.Tensor,
                   codebook: torch.Tensor, group_size: int) -> torch.Tensor:
    """Fused mat-vec against an E8-lattice-coded weight matrix.

    Args:
        rx: Rotated activations ``(K,)`` float32.
        words: Packed 16-bit lattice codes as int32 words ``(N, K/16)``; each
            word holds two codes, each naming one 8-dimensional lattice point,
            so a word covers 16 input coordinates exactly as the 2-bit scalar
            path does.
        scale: Per-group scales ``(N, K/group_size)`` float32.
        codebook: Lattice points ``(2**16, 8)`` float16 or float32.
        group_size: Scale group size.

    Returns:
        Output ``(1, N)`` float32.
    """
    if not GEMV_E8_AVAILABLE:
        raise RuntimeError("triton is not available")
    K = rx.shape[-1]
    N, words_per_row = words.shape
    D = codebook.shape[1]
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
    _gemv_e8_kernel[(n_blocks, split_k)](
        rx, words, scale, codebook, out,
        N, K, words_per_row, words_per_chunk,
        words.stride(0), scale.stride(0), scale.stride(1),
        GROUP_SIZE=group_size, SPLIT_K=split_k,
        BLOCK_N=block_n, BLOCK_W=block_w, D=D,
        num_warps=4, num_stages=3,
    )
    return out
