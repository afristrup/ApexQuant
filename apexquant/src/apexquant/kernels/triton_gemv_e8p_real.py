"""Fused GEMV against the *actual* E8P construction, not a decode-shape proxy.

`triton_gemv_e8p` measures the decode cost of a code with E8P's *shape*: an
8-bit index into a small resident source table plus 8 free sign bits. It says so
in its own docstring, and the paper's appendix repeats the caveat, because that
code is not E8P: it omits the parity inference and the +/-1/4 shift, and its
table is 4 KiB rather than 1 KiB.

That was defensible while E8P was only a row in a table. It is not defensible now
that we ship E8P artifacts, so this kernel decodes the real format, the one
`apexquant.vq.e8p_codec` emits:

    bits  0..7   j, an entry of the 256-word source codebook S = |D8h|
    bits  8..14  the signs of coordinates 0..6, one bit each (1 = negative)
    bit   15     the shift c: 0 -> +1/4, 1 -> -1/4

Coordinate 7's sign is not stored. It is recovered from the parity constraint
that puts the codeword in D8h: the number of negated coordinates must have
parity `par[j]`. So the decode costs three things the proxy did not pay: a
population count over seven bits, one extra integer op to recover the last sign,
and a fused add for the shift. Whether those are free at batch one, where the
decode is bandwidth-bound, is the question this kernel exists to answer, and it
can now be answered about E8P rather than about something shaped like it.

Resident state is 256 x 8 magnitudes plus 256 parity bits. In fp16 that is
4 KiB + 256 B here; E8P proper packs the magnitudes as 4-bit integers for 1 KiB,
which we do not do, so this measures the L1-resident regime without claiming
E8P's exact footprint.
"""

from __future__ import annotations

import functools

import torch

try:
    import triton
    import triton.language as tl

    GEMV_E8P_REAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    GEMV_E8P_REAL_AVAILABLE = False


if GEMV_E8P_REAL_AVAILABLE:

    @triton.jit
    def _gemv_e8p_real_kernel(
        rx_ptr, words_ptr, scale_ptr, src_ptr, par_ptr, out_ptr,
        shift_mag,
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
            # two 16-bit codes per 32-bit word, each naming one D-dim codeword
            for h in tl.static_range(2):
                code = (word >> (16 * h)) & 0xFFFF          # (BN, BW)
                j = code & 0xFF                             # source entry
                s7 = (code >> 8) & 0x7F                     # signs of coords 0..6
                cbit = (code >> 15) & 1                     # shift selector

                # magnitudes: 4 KiB table, stays in L1
                sp = src_ptr + j[:, :, None] * D + offs_d[None, None, :]
                base = tl.load(sp).to(tl.float32)           # (BN, BW, D)

                # stored sign bits cover coordinates 0..6 only
                stored = (s7[:, :, None] >> offs_d[None, None, :]) & 1
                stored = tl.where(offs_d[None, None, :] < D - 1, stored, 0)
                # population count over the seven stored bits
                nlo = tl.sum(stored, axis=2)                # (BN, BW)
                # recover coordinate D-1: total negations must match par[j].
                # last = (par - nlo) mod 2 == (par + nlo) & 1, since -1 = 1 mod 2.
                par = tl.load(par_ptr + j)                  # (BN, BW)
                last = (par + nlo) & 1
                neg = tl.where(offs_d[None, None, :] < D - 1,
                               stored, last[:, :, None])
                sgn = 1.0 - 2.0 * neg.to(tl.float32)

                # The codec's globally tuned scale is folded into `src` by the
                # caller, so it must be folded into the shift too: the codeword
                # is (sgn*S + c)*kappa == sgn*(S*kappa) + c*kappa. Omitting it
                # from the shift alone leaves the output off by 1/kappa.
                shift = tl.where(cbit == 1, -shift_mag, shift_mag).to(tl.float32)
                vec = sgn * base + shift[:, :, None]

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


def triton_gemv_e8p_real(rx: torch.Tensor, words: torch.Tensor,
                         scale: torch.Tensor, source: torch.Tensor,
                         parity: torch.Tensor, group_size: int,
                         codec_scale: float = 1.0) -> torch.Tensor:
    """Fused mat-vec against real E8P codes.

    Args:
        rx: Rotated activations ``(K,)`` float32.
        words: E8P codes packed two-per-int32-word, ``(N, K/16)``. Identical
            weight bandwidth to the scalar 2-bit and stored-table arms, so a
            timing difference is decode cost alone.
        scale: Per-group scales ``(N, K/group_size)`` float32.
        source: The ``(256, 8)`` magnitude table ``S = |D8h|``, *unscaled*.
        parity: ``(256,)`` int32 required flip parity per source entry.
        group_size: Scale group size.
        codec_scale: The codec's single globally tuned scale ``kappa``. Folded
            into the source table and the shift here rather than applied per
            element, so it costs nothing at run time. Leaving it at 1.0 when the
            codec has a real scale makes every output wrong by ``1/kappa``.

    Returns:
        Output ``(1, N)`` float32.
    """
    if not GEMV_E8P_REAL_AVAILABLE:
        raise RuntimeError("triton is not available")
    K = rx.shape[-1]
    N, words_per_row = words.shape
    D = source.shape[1]
    vals_per_word = 2 * D
    if words_per_row * vals_per_word != K:
        raise ValueError(
            f"words ({words_per_row} x {vals_per_word}) do not cover K={K}")
    if group_size % vals_per_word:
        raise ValueError(f"group_size {group_size} must be a multiple of "
                         f"{vals_per_word}")

    out = torch.zeros(N, device=rx.device, dtype=torch.float32)
    BLOCK_N, BLOCK_W = 8, 8
    # split-K over words so every SM gets work at batch one, where a mat-vec
    # would otherwise leave most of the GPU idle
    n_blocks = triton.cdiv(N, BLOCK_N)
    target = max(1, _sm_count(rx.device.index or 0) // max(1, n_blocks))
    chunks = min(target, triton.cdiv(words_per_row, BLOCK_W))
    words_per_chunk = triton.cdiv(words_per_row, chunks) if chunks else words_per_row

    src = (source * codec_scale).contiguous()
    _gemv_e8p_real_kernel[(n_blocks, chunks)](
        rx, words, scale, src, parity, out,
        0.25 * codec_scale,
        N, K, words_per_row, words_per_chunk,
        words.stride(0), scale.stride(0), scale.stride(1),
        GROUP_SIZE=group_size, BLOCK_N=BLOCK_N, BLOCK_W=BLOCK_W, D=D,
    )
    return out.view(1, N)
