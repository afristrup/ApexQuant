"""Fused GEMV against the *shipped* trellis code, the one ``TrellisCodec`` emits.

Two trellis kernels already exist and neither is this. :mod:`triton_gemv_tcq` is
a fused mat-vec for a different code -- ``L=16``, tail-biting inside each 32-bit
word, a 1MAD-computed codeword, no Viterbi -- and says so in its own docstring;
it exists to price a lookup-free code at matched bandwidth, not to decode our
checkpoints. ``TrellisCodec._walk_kernel`` decodes the right code but is a
*dequantiser*: it writes every reconstructed weight to DRAM for cuBLAS to read
back. This is the missing third thing -- the real code, consumed inside a
mat-vec, with the dense weight never materialised.

What makes it possible is the property the walk's own docstring states: the
register is ``L`` bits wide and eats ``k`` per step, and ``TrellisCodec``
requires ``k`` to divide ``L``, so after ``M = L/k`` steps no earlier symbol
survives. Weight ``t`` is a pure function of the ``M`` symbols ending at ``t``.
There is no recurrence to walk, only a window to extract, and windows are
independent.

Two facts make the extraction free rather than merely possible:

* **The window is contiguous in the packed stream.** ``pack_indices`` puts
  symbol ``p`` at bit offset ``k * (p % per)`` of byte ``p // per``, so reading
  four bytes little-endian into an int32 word puts positions in increasing bit
  order. The ``M`` symbols ending at ``t`` are then one contiguous ``L``-bit
  field, possibly straddling the word boundary.

* **That field holds the state's symbols reversed.** The walk builds
  ``state = (state << k) | sym_t``, so the newest symbol lands in the LOW bits;
  the packed stream has it in the HIGH ones. Reversing ``M`` fields per weight
  inside the kernel would cost more than the decode it replaces, so we reverse
  once on the host and permute the codeword table instead -- see
  :func:`trellis_window_table`. The kernel indexes the permuted table with the
  raw window and pays nothing.

Group boundaries are free too. ``progressive_artifact`` encodes
``(out * groups, gs)``, one row per ``(output row, scale group)``, so the
trellis restarts at state zero at every scale group. Reading the preceding word
as zero there reproduces the walk's zero padding exactly, which is why the
previous-word load is masked on ``offs_w % WORDS_PER_GROUP`` instead of being
carried in a register across the loop -- the mask also makes split-K chunk
boundaries a non-issue, since a chunk that starts mid-group still reads its true
predecessor.

The blocking and split-K policy are copied from :mod:`triton_gemv_tcq`, which
copied them from :mod:`triton_gemv_mp`, so against those arms the only
difference is the arithmetic between loading a word and multiplying by an
activation.

All ``P`` residual passes run in ONE launch. The grid is
``(cdiv(N, BLOCK_N), SPLIT_K * P)``: each program reduces one
``(rows x word-chunk)`` tile of one pass and writes its partial sum to its own
row of a ``(SPLIT_K * P, N)`` buffer, so a pass costs no more machinery than a
K-chunk does. The per-pass path would instead pay ``P`` rotations, ``P``
launches and ``P`` output read-modify-writes per layer.

Each program owning a row, rather than every program accumulating into the
output with ``atomic_add``, is what makes the result reproducible. Atomics add
the partials in whatever order the programs retire, and fp32 addition is not
associative, so the same call returned different bits from one run to the next:
at ``N=4096, K=14336, P=3``, 19 of 20 repeats differed from the first by up to
4e-06 absolute. That is a defensible error bar and an indefensible thing to put
under a perplexity comparison, where the quantity read off is a difference
between two runs. The buffer costs at most ``2 * SMs * BLOCK_N * P`` floats,
since ``SPLIT_K`` is sized against the SM count and not against ``K``, and
folding it costs one small launch: about 1.5 us on a 4090 at ``N=4096``, which
is 7% at ``K=4096, P=1`` (0.0207 -> 0.0222 ms), 0.3% at ``K=14336, P=3``, and
negative at ``K=4096, P=3`` (0.0545 -> 0.0529 ms), where losing the atomics and
the output memset more than pays for it.

One table serves every pass, which is Proposition 2 doing real work rather than
decorating a proof: a fresh Haar rotation returns the residual to the same
spherical marginal, so the same codec -- and therefore the same permuted
codeword table -- is optimal at every pass. A method that re-fitted its codebook
per pass would need ``P`` tables resident, and this kernel's cost is dominated by
exactly that residency (see ``experiments/trellis_real_kernel_check.py``).
"""

from __future__ import annotations

import functools
import os

import torch

from .fold_partials import fold_partials

try:
    import triton
    import triton.language as tl

    GEMV_TRELLIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    GEMV_TRELLIS_AVAILABLE = False


if GEMV_TRELLIS_AVAILABLE:

    @triton.jit
    def _gemv_trellis_kernel(
        rx_ptr, words_ptr, scale_ptr, cw_ptr, part_ptr,
        N, K, words_per_row, words_per_chunk,
        stride_rp, stride_wp, stride_wn, stride_sp, stride_sn, stride_sg,
        GROUP_SIZE: tl.constexpr, WORDS_PER_GROUP: tl.constexpr,
        BITS: tl.constexpr, L: tl.constexpr, VPW: tl.constexpr,
        SPLIT_K: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_z = tl.program_id(1)
        p = pid_z // SPLIT_K             # residual pass
        chunk = pid_z % SPLIT_K          # word-chunk within the pass
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
            cur = tl.load(wrd_base + offs_w[None, :],
                          mask=n_mask[:, None] & w_mask[None, :], other=0)
            # The preceding word, or zero at a group boundary -- which is exactly
            # the walk's zero padding, because a group starts the trellis at
            # state zero. The mask is also what keeps the offs_w == 0 address
            # from being read.
            has_prev = w_mask & ((offs_w % WORDS_PER_GROUP) != 0)
            prev = tl.load(wrd_base + offs_w[None, :] - 1,
                           mask=n_mask[:, None] & has_prev[None, :], other=0)

            wsum = tl.zeros((BLOCK_N, BLOCK_W), dtype=tl.float32)
            for v in tl.static_range(VPW):
                # Bit range of the window for position v, measured from the low
                # bit of `cur`. Negative means it reaches back into `prev`.
                # lo + L <= 32 always, since lo peaks at 32 - L when v = VPW - 1.
                lo = BITS * v - (L - BITS)
                if lo >= 0:
                    win = (cur >> lo) & ((1 << L) - 1)
                else:
                    # `>>` on int32 is arithmetic, but the mask is exactly as
                    # wide as the bits that survive the shift, so the sign
                    # extension is discarded rather than merely unlikely.
                    top = -lo                            # bits taken from prev
                    win = (((prev >> (32 + lo)) & ((1 << top) - 1))
                           | ((cur & ((1 << (L - top)) - 1)) << top))
                code = tl.load(cw_ptr + win)
                kk = offs_w * VPW + v
                rxv = tl.load(rx_base + kk, mask=kk < K, other=0.0)
                wsum += code * rxv[None, :]
            g = (offs_w * VPW) // GROUP_SIZE  # a word lies in one scale group
            sc = tl.load(sc_base + g[None, :] * stride_sg,
                         mask=n_mask[:, None] & w_mask[None, :], other=0.0)
            acc += tl.sum(wsum * sc, axis=1)
        # One row per program, contiguous, for the caller to fold in a fixed
        # order. An atomic reduction into the output would save that fold and
        # spend the result's reproducibility on it.
        tl.store(part_ptr + pid_z * N + offs_n, acc, mask=n_mask)


# A window table carries one entry past its ``2**L`` codewords, and that entry is
# the only thing separating it from ``codec.codeword``: same length otherwise,
# same dtype, same device, and a decode that is silently field-reversed. The
# length is what the kernels check, because reading it costs nothing on a decode
# path where a value check would cost a device sync per layer. The value is NaN
# so that a read past the codewords is loud as well.
    @triton.jit
    def _gemv_trellis_fused_pass_kernel(
        rx_ptr, words_ptr, scale_ptr, cw_ptr, out_ptr,
        N, K, words_per_row, words_per_chunk,
        stride_rp, stride_wp, stride_wn, stride_sp, stride_sn, stride_sg,
        GROUP_SIZE: tl.constexpr, WORDS_PER_GROUP: tl.constexpr,
        BITS: tl.constexpr, L: tl.constexpr, VPW: tl.constexpr,
        SPLIT_K: tl.constexpr, PASSES: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_z = tl.program_id(1)
        chunk = pid_z
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        w_lo = chunk * words_per_chunk
        w_hi = tl.minimum(w_lo + words_per_chunk, words_per_row)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for p in range(PASSES):
            active = True
            rx_base = rx_ptr + p * stride_rp
            wrd_base = words_ptr + p * stride_wp + offs_n[:, None] * stride_wn
            sc_base = scale_ptr + p * stride_sp + offs_n[:, None] * stride_sn
            for w0 in range(w_lo, w_hi, BLOCK_W):
                offs_w = w0 + tl.arange(0, BLOCK_W)
                w_mask = (offs_w < w_hi) & active
                cur = tl.load(wrd_base + offs_w[None, :], mask=n_mask[:, None] & w_mask[None, :], other=0)
                has_prev = w_mask & ((offs_w % WORDS_PER_GROUP) != 0)
                prev = tl.load(wrd_base + offs_w[None, :] - 1, mask=n_mask[:, None] & has_prev[None, :], other=0)
                wsum = tl.zeros((BLOCK_N, BLOCK_W), dtype=tl.float32)
                for v in tl.static_range(VPW):
                    lo = BITS * v - (L - BITS)
                    if lo >= 0:
                        win = (cur >> lo) & ((1 << L) - 1)
                    else:
                        top = -lo
                        win = (((prev >> (32 + lo)) & ((1 << top) - 1)) | ((cur & ((1 << (L - top)) - 1)) << top))
                    code = tl.load(cw_ptr + win, mask=w_mask[None, :], other=0)
                    kk = offs_w * VPW + v
                    rxv = tl.load(rx_base + kk, mask=(kk < K) & active, other=0.0)
                    wsum += code * rxv[None, :]
                g = (offs_w * VPW) // GROUP_SIZE
                sc = tl.load(sc_base + g[None, :] * stride_sg, mask=n_mask[:, None] & w_mask[None, :], other=0.0)
                acc += tl.sum(wsum * sc, axis=1)
        tl.store(out_ptr + pid_z * N + offs_n, acc, mask=n_mask)


WINDOW_TABLE_TAG = float("nan")


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def trellis_window_table(codec) -> torch.Tensor:
    """Codeword table re-indexed by the packed window, codec scale folded in.

    The walk indexes ``codec.codeword`` by ``state``, whose ``k``-bit fields hold
    the last ``M`` symbols newest-first. The kernel extracts the raw window from
    the packed stream, which holds the same symbols oldest-first. Field ``j`` of
    the window is the symbol at ``t - (M - 1 - j)``, which the walk puts in field
    ``M - 1 - j`` of the state, so the two differ by a reversal of ``M`` fields
    and nothing else. Doing it here costs one pass over ``2**L`` entries per
    codec; doing it in the kernel would cost ``M`` shifts per weight.

    Folding ``codec.scale`` in as well matches ``_walk_kernel``, which also
    applies it at the point of the table read, so the products are bit-identical
    rather than merely equal.

    Args:
        codec: A :class:`~apexquant.vq.trellis_codec.TrellisCodec`.

    Returns:
        Float32 ``(2**trellis_bits + 1,)`` table for the trellis kernels: the
        ``2**L`` permuted codewords, then :data:`WINDOW_TABLE_TAG`. The kernels
        never index the last entry; it is there so that ``table.numel()``
        identifies a table this function built. See :func:`check_window_table`.
    """
    k, L = codec.bits, codec.trellis_bits
    M = L // k
    win = torch.arange(1 << L, device=codec.device, dtype=torch.int64)
    state = torch.zeros_like(win)
    for j in range(M):
        state |= ((win >> (k * j)) & ((1 << k) - 1)) << (k * (M - 1 - j))
    cw = (codec.codeword[state] * codec.scale).to(torch.float32)
    tag = torch.full((1,), WINDOW_TABLE_TAG, dtype=torch.float32,
                     device=cw.device)
    return torch.cat((cw, tag)).contiguous()


def check_window_table(table: torch.Tensor, trellis_bits: int) -> None:
    """Fail before the launch if the table is not a window table.

    ``codec.codeword`` is indexed by the walk's state, which is the packed
    window with its ``M`` symbol fields reversed. It is the same length, dtype
    and device as a window table, so a kernel handed it decodes a reversed code
    at full speed and returns weights that are wrong everywhere -- the failure
    the extra entry from :func:`trellis_window_table` exists to make loud.

    Args:
        table: The table a caller passed to a trellis kernel.
        trellis_bits: Shift-register width ``L``.

    Raises:
        ValueError: If ``table`` did not come from :func:`trellis_window_table`.
    """
    if table.numel() != (1 << trellis_bits) + 1:
        raise ValueError(
            f"table has {table.numel()} entries, not the "
            f"{(1 << trellis_bits) + 1} that trellis_window_table(codec) "
            f"returns at trellis_bits {trellis_bits}. Passing codec.codeword "
            "decodes a field-reversed code, so it is rejected rather than run.")


def triton_gemv_trellis(rx: torch.Tensor, words: torch.Tensor,
                        scale: torch.Tensor, table: torch.Tensor,
                        group_size: int, bits: int,
                        trellis_bits: int) -> torch.Tensor:
    """Fused mat-vec against real bitshift-trellis codes, all passes in one launch.

    Args:
        rx: Rotated activations ``(P, K)`` float32, pass ``p`` rotated by the
            seed the encoder used for it. ``P = 1`` is a leading axis of one, not
            a separate code path.
        words: Packed symbols as int32 words ``(P, N, K * bits // 32)`` -- the
            codec's uint8 output viewed as int32, so the weight bandwidth is the
            stored bandwidth exactly.
        scale: Per-group scales ``(P, N, K // group_size)`` float32.
        table: ``(2**trellis_bits + 1,)`` from :func:`trellis_window_table`,
            shared by every pass. Anything else, ``codec.codeword`` above all,
            is rejected by :func:`check_window_table`.
        group_size: Scale group size, and the length of one trellis run.
        bits: Symbol width ``k``.
        trellis_bits: Shift-register width ``L``; must be a multiple of ``bits``.

    Returns:
        Output ``(1, N)`` float32, summed over passes.
    """
    if not GEMV_TRELLIS_AVAILABLE:
        raise RuntimeError("triton is not available")
    if trellis_bits % bits:
        raise ValueError(f"trellis_bits {trellis_bits} must be a multiple of "
                         f"bits {bits}")
    check_window_table(table, trellis_bits)
    if trellis_bits >= 32:
        raise ValueError(f"trellis_bits {trellis_bits} must be under 32; the "
                         "window is extracted from a pair of int32 words")
    if rx.ndim != 2 or words.ndim != 3 or scale.ndim != 3:
        raise ValueError("rx must be (P, K), words (P, N, W), scale (P, N, G); "
                         f"got {tuple(rx.shape)}, {tuple(words.shape)}, "
                         f"{tuple(scale.shape)}")
    P, K = rx.shape
    if words.shape[0] != P or scale.shape[0] != P:
        raise ValueError(f"pass counts disagree: rx {P}, words {words.shape[0]}, "
                         f"scale {scale.shape[0]}")
    _, N, words_per_row = words.shape
    vpw = 32 // bits
    if group_size % vpw:
        raise ValueError(f"group_size {group_size} must be a multiple of {vpw}")
    if words_per_row * vpw != K:
        raise ValueError(
            f"words ({words_per_row} x {vpw}) do not cover K={K}")
    wpg = group_size // vpw  # words per scale group, and per trellis run

    large_output = N >= 16000
    default_block_n = 32 if large_output else 64
    block_n = int(os.environ.get("APEXQUANT_GEMV_BLOCK_N", default_block_n))
    block_w = int(os.environ.get("APEXQUANT_GEMV_BLOCK_W", "16"))
    default_warps = 2 if large_output or (N <= 12288 and K <= 4096) else 4
    warps = int(os.environ.get("APEXQUANT_GEMV_WARPS", default_warps))
    stages = int(os.environ.get("APEXQUANT_GEMV_STAGES", "3"))
    if block_n not in (32, 64, 128) or block_w not in (16, 32, 64):
        raise ValueError("GEMV block sizes must be BLOCK_N in {32, 64, 128} "
                         "and BLOCK_W in {16, 32, 64}")
    if warps not in (2, 4, 8) or stages not in (2, 3, 4):
        raise ValueError("invalid GEMV warps or stages")
    n_blocks = triton.cdiv(N, block_n)
    sms = _sm_count(rx.device.index or 0)
    max_chunks = max(1, words_per_row // wpg)
    # P does not enter here, and dividing by it is a trap worth naming: the
    # passes multiply the program count on their own, so sizing the split to a
    # target program count collapses it to 1 at P=3 and makes each program walk
    # the whole row serially. Measured, that cost 25-45% and turned the fusion
    # into a loss against separate launches. What the split actually bounds is
    # the serial walk per program, and that bound does not depend on P.
    requested_split = int(os.environ.get("APEXQUANT_GEMV_SPLIT_K", "0"))
    split_k = (requested_split if requested_split else
               max(1, (2 * sms) // max(1, n_blocks)))
    if not requested_split and N <= 4096:
        split_k = min(split_k, 2)
    split_k = min(max(1, split_k), max_chunks)
    words_per_chunk = triton.cdiv(triton.cdiv(words_per_row, split_k), wpg) * wpg

    fused_passes = split_k == 1 or os.environ.get("APEXQUANT_GEMV_FUSED_PASSES", "0") == "1"
    if fused_passes:
        parts = torch.empty((split_k, N), dtype=torch.float32, device=rx.device)
        _gemv_trellis_fused_pass_kernel[(n_blocks, split_k)](
            rx, words, scale, table, parts,
            N, K, words_per_row, words_per_chunk,
            rx.stride(0), words.stride(0), words.stride(1),
            scale.stride(0), scale.stride(1), scale.stride(2),
            GROUP_SIZE=group_size, WORDS_PER_GROUP=wpg,
            BITS=bits, L=trellis_bits, VPW=vpw,
            SPLIT_K=split_k, PASSES=P,
            BLOCK_N=block_n, BLOCK_W=block_w,
            num_warps=warps, num_stages=stages,
        )
        if split_k == 1:
            return parts
        return fold_partials(parts, split_k, 1, N)

    parts = torch.empty((split_k * P, N), dtype=torch.float32,
                        device=rx.device)
    _gemv_trellis_kernel[(n_blocks, split_k * P)](
        rx, words, scale, table, parts,
        N, K, words_per_row, words_per_chunk,
        rx.stride(0), words.stride(0), words.stride(1),
        scale.stride(0), scale.stride(1), scale.stride(2),
        GROUP_SIZE=group_size, WORDS_PER_GROUP=wpg,
        BITS=bits, L=trellis_bits, VPW=vpw,
        SPLIT_K=split_k,
        BLOCK_N=block_n, BLOCK_W=block_w,
        num_warps=warps, num_stages=stages,
    )
    # The fixed reduction order is the point: this is what the kernel's atomics
    # used to do in whatever order the programs retired in.
    if parts.shape[0] == 1:
        return parts  # see the GEMM's fold: a length-one sum is a copy
    return fold_partials(parts, parts.shape[0], 1, N)
