"""Multi-row fused GEMM against the shipped trellis code, all passes in one launch.

:mod:`triton_gemv_trellis` is a mat-vec, so anything wider than one row had to
reconstruct the layer and call cuBLAS -- correct, and bounded in memory since the
reconstruction is per layer and transient, but it puts a dense fp32 weight
through DRAM for every prefill token block. This is the tile version, and it is
what makes a perplexity measurement through the packed path practical rather than
merely possible.

The structure is ``triton_gemm_mid``'s, which is the scalar stage's answer to the
same problem: pass index and a group-aligned K-split join the grid so the device
fills at small M, ``BLOCK_K`` divides the group size so a K-tile lies in exactly
one scale group and the scale is one vector load per tile, and the whole thing is
one launch for every pass. Only the decode differs.

The join over passes and K-chunks is a write, not an ``atomic_add``: each program
owns its tile of a ``(SPLIT_K * P * M, N)`` buffer, one ``M``-row band per grid
z, and the caller folds the bands in a fixed order. Atomics would add the partials in whatever order the programs retire, and
fp32 addition is not associative, so the same call could return different bits
from one run to the next -- observed at ``M=1``, and a thing no perplexity
comparison should have to carry. It costs ``P`` output copies of scratch, since
``SPLIT_K`` is 1 at any prefill that fills the device: 60 MiB against 15 at the
widest dispatched shape, transient, and small beside the 236 MiB the
reconstruction fallback above ``TRELLIS_GEMM_MAX_ROWS`` allocates. The fold
itself is 5 us at ``M=8`` and 17 us at ``M=1024``, against 0.50 and 7.4 ms of
kernel. Time overall it is close to free and mostly a gain, once the small tile
is given the warps it now needs: against the atomics it is 0.62 -> 0.50 ms at
``M=1``, 0.65 -> 0.51 at ``M=32``, 1.11 -> 1.08 at ``M=128``, level at ``M=512``,
and 2.6% dearer only at ``M=2, 4, 8`` (0.492 -> 0.505), where the atomic form
happened to hit its one fast tile.

The decode differs in one way that matters. In the mat-vec the position within a
word is a ``tl.static_range`` index, so the window's bit offset is a compile-time
constant and the straddle across the word boundary compiles to two shifts of
known width. Here a K-tile covers ``BLOCK_K`` consecutive coordinates at once, so
that position is a runtime value and the offset is a vector. Rather than
reintroduce a branch per element, the two words are concatenated into one int64
and the window is a single variable shift:

    pair = prev | (cur << 32)
    win  = (pair >> (32 + k*v - (L - k))) & (2^L - 1)

The shift is in ``[32 - (L - k), 64 - L]`` for every ``v``, so it never goes
negative and never runs off the top, and the mask discards the arithmetic shift's
sign extension the same way the mat-vec's does. int64 costs more per element than
the two int32 shifts it replaces, which would matter if this kernel were bound by
arithmetic; it is bound by the codeword gather, the same as the mat-vec.

Which is what ``BLOCK_M`` is for. A decoded weight tile serves ``BLOCK_M`` rows,
so doubling it halves the gathers per unit of output, and that is the whole
tuning story: on a 4090 at ``N=3840, K=15360, P=3``, moving from the
``triton_gemm_mid`` defaults to ``BLOCK_M=BLOCK_N=128`` with eight warps is
1.58x at both ``M=512`` (5.90 -> 3.73 ms) and ``M=2048`` (23.4 -> 14.9 ms).

**This kernel cannot beat reconstructing the layer at large M, and no amount of
tuning changes that.** The fused form computes ``sum_p rx_p @ W_p^T``, which is
``P`` dots of ``2MNK``; reconstruction pays ``O(NK)`` once and then does one. The
extra ``(P-1) * 2MNK`` grows with M while the reconstruction does not, so there
is a crossover, and it is structural rather than a tuning artefact.
:class:`ProgressiveTrellisLinear` dispatches on it -- see
:data:`TRELLIS_GEMM_MAX_ROWS`.

The passes cannot be summed into one weight before the dot, which is what would
remove the factor: each pass carries its own rotation, and undoing those is
exactly the ``O(NK * group_size)`` work that reconstruction does. Sharing one
rotation across passes would collapse it, and would also destroy the
re-isotropisation the method depends on.
"""

from __future__ import annotations

import functools

import torch

from .fold_partials import fold_partials
from .triton_gemv_trellis import check_window_table

try:
    import triton
    import triton.language as tl

    GEMM_TRELLIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    GEMM_TRELLIS_AVAILABLE = False


if GEMM_TRELLIS_AVAILABLE:

    # 1 / sqrt(2), the argument scale of the exact (erf) GELU.
    INV_SQRT2 = tl.constexpr(0.7071067811865476)

    # Which tile wins is not something a rule predicts here. Measured at Clay's
    # four shapes: qkv (N=3072, K=1024) and ff1 (N=4096, K=1024) share K and want
    # opposite tiles; out and ff2 both have N=1024 and want opposite tiles. And
    # the tile that is best across Clay costs gemma-3 15%. So the choice is left
    # to the autotuner, which also means a model this was never measured on gets
    # a tuned kernel instead of Clay's leftovers.
    GEMM_CONFIGS = [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                      num_warps=warps, num_stages=stages)
        for bm, bn, bk, warps, stages in [
            # Every entry is here because some real shape picked it out of a
            # 72-config sweep: Clay's four linears and its patch embedding at
            # P=1 and P=2, and the gemma-3 shape at P=3. Widening the sweep
            # beyond this bought P=2 nothing further and P=1 nothing at all.
            (64, 32, 16, 4, 2),
            (64, 32, 32, 4, 1),
            (64, 32, 32, 8, 1),
            (64, 32, 32, 8, 2),
            (64, 64, 16, 4, 1),
            (64, 64, 16, 4, 2),
            (64, 64, 32, 4, 1),
            (64, 64, 32, 4, 2),
            (64, 64, 32, 8, 1),
            (128, 32, 32, 4, 2),
            (128, 32, 32, 8, 2),
            (128, 64, 16, 4, 2),
            (128, 64, 32, 4, 2),
            (128, 128, 16, 4, 1),
            (128, 128, 16, 4, 2),
            (128, 128, 32, 4, 1),
            (128, 128, 32, 8, 2),
        ]
    ]

    def prune_gemm_configs(configs, named_args, **kwargs):
        """Drop tiles whose K block cannot address the code stream.

        A K tile has to lie inside one scale group and hold whole packed words,
        so ``BLOCK_K`` must divide ``GROUP_SIZE`` and be a multiple of the values
        per word. Without this the autotuner would happily benchmark a config
        that decodes the wrong bits, and pick it if it were fast.
        """
        group_size = kwargs["GROUP_SIZE"]
        per_word = kwargs["VPW"]
        keep = [c for c in configs
                if group_size % c.kwargs["BLOCK_K"] == 0
                and c.kwargs["BLOCK_K"] % per_word == 0]
        return keep or [configs[0]]

    @triton.autotune(configs=GEMM_CONFIGS, key=["M", "N", "K", "SPLIT_K"],
                     prune_configs_by={"early_config_prune": prune_gemm_configs})
    @triton.jit
    def _gemm_trellis_kernel(
        rx_ptr, words_ptr, scale_ptr, cw_ptr, part_ptr, bias_ptr,
        M, N, K, words_per_chunk,
        stride_rp, stride_rm,             # rx (P, M, K), K contiguous
        stride_wp, stride_wn,             # words (P, N, W)
        stride_sp, stride_sn, stride_sg,  # scale (P, N, G)
        stride_om, stride_on,             # partials ((SPLIT_K * P) * M, N)
        GROUP_SIZE: tl.constexpr, WORDS_PER_GROUP: tl.constexpr,
        BITS: tl.constexpr, L: tl.constexpr, VPW: tl.constexpr,
        SPLIT_K: tl.constexpr, PREC: tl.constexpr,
        HAS_BIAS: tl.constexpr, ACT_GELU: tl.constexpr,
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
        part_base = part_ptr + (pid_z * M + offs_m)[:, None] * stride_om

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(k_lo, k_hi, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            w_idx = offs_k // VPW                 # word holding this coordinate
            shift = 32 + BITS * (offs_k % VPW) - (L - BITS)

            # Chunks are group-aligned and BLOCK_K divides both GROUP_SIZE and
            # the chunk length, so the tile is full and lies in one group.
            rx = tl.load(rx_base + offs_k[None, :],
                         mask=m_mask[:, None], other=0.0)
            cur = tl.load(wrd_base + w_idx[None, :],
                          mask=n_mask[:, None], other=0)
            # Zero at a group boundary, which is the walk's zero padding: a scale
            # group is one trellis run and starts the register at zero.
            has_prev = (w_idx % WORDS_PER_GROUP) != 0
            prev = tl.load(wrd_base + w_idx[None, :] - 1,
                           mask=n_mask[:, None] & has_prev[None, :], other=0)

            pair = (cur.to(tl.int64) << 32) | (prev.to(tl.int64) & 0xFFFFFFFF)
            win = ((pair >> shift[None, :]) & ((1 << L) - 1)).to(tl.int32)
            code = tl.load(cw_ptr + win)

            sc = tl.load(sc_base + (k0 // GROUP_SIZE) * stride_sg,
                         mask=n_mask, other=0.0)          # one group per tile
            acc += tl.dot(rx, tl.trans(code * sc[:, None]), input_precision=PREC)

        # Epilogue. Only reachable when this program owns the whole output
        # element, which the launcher enforces: with more than one band the
        # bias would be added once per band and the activation applied to a
        # partial sum. Fusing it saves a full read-modify-write of the output,
        # which at Clay's expanding projection is 25 MB of traffic a layer.
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        if ACT_GELU:
            acc = 0.5 * acc * (1.0 + tl.math.erf(acc * INV_SQRT2))

        # One tile per program, for the caller to fold in a fixed order,
        # exactly as the mat-vec does: an atomic reduction here would spend the
        # result's reproducibility to save that fold.
        tl.store(part_base + offs_n[None, :] * stride_on, acc,
                 mask=m_mask[:, None] & n_mask[None, :])


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


#: Up to this many rows the packed GEMM beats reconstructing the layer; past it
#: the (P-1) extra dots cost more than the reconstruction they avoid. Measured on
#: a 4090 at P=3 over two real gemma-3 layers, which agree on where it falls --
#: packed wins at M=1024 (7.49 ms vs 9.42 on down_proj, 1.97 vs 2.72 on q_proj)
#: and loses at M=2048 (14.96 vs 11.88, 4.03 vs 3.24). The margin is much wider
#: below: 7x at M=128. It is a ratio of arithmetic to a fixed cost, so it moves
#: with P and with the shape; a serviceable default, not a constant of nature.
TRELLIS_GEMM_MAX_ROWS = 1024


def trellis_gemm_block_k(group_size: int, bits: int, prefer: int = 64) -> int:
    """Largest supported K-tile up to ``prefer``, or 0 if there is no GEMM here.

    A tile must lie in one scale group and hold whole words, so ``BLOCK_K`` has
    to divide ``GROUP_SIZE`` and be a multiple of ``32 // bits``. Every real
    layer clears this; the check exists so a narrow one falls back rather than
    decoding the wrong thing.
    """
    vpw = 32 // bits
    for block_k in (128, 64, 32, 16):
        if (block_k <= prefer and block_k >= vpw and block_k % vpw == 0
                and group_size % block_k == 0):
            return block_k
    return 0


def triton_gemm_trellis(rx: torch.Tensor, words: torch.Tensor,
                        scale: torch.Tensor, table: torch.Tensor,
                        group_size: int, bits: int,
                        trellis_bits: int,
                        precision: str = "tf32",
                        bias: torch.Tensor | None = None,
                        gelu: bool = False) -> torch.Tensor:
    """All-passes fused GEMM: ``out[m,n] = sum_p sum_k rx[p,m,k] * w_p[n,k]``.

    Args:
        rx: Rotated activations ``(P, M, K)`` float32, K contiguous.
        words: Packed symbols as int32 words ``(P, N, K*bits/32)``.
        scale: Per-group scales ``(P, N, K/group_size)`` float32.
        table: ``(2**trellis_bits + 1,)`` from ``trellis_window_table``. The
            table is the mat-vec's, and so is the check that rejects
            ``codec.codeword``, which would decode field-reversed.
        group_size: Scale group size, and the length of one trellis run.
        bits: Symbol width ``k``.
        trellis_bits: Shift-register width ``L``.
        precision: ``tl.dot`` input precision. Measured against an fp32 dense
            reference on the shape in the module docstring:

            ====== ============ ===================================
            mode   max rel err  cost vs tf32, at the tiles below
            ====== ============ ===================================
            tf32     9e-04        1.0x
            tf32x3   1e-06        1.8x at M=2, 40x at M=64 and up
            ieee     5e-06        56x everywhere
            ====== ============ ===================================

            Neither alternative is usable. ``ieee`` drops off the tensor cores
            entirely. ``tf32x3`` is affordable only on the smallest tile and
            collapses on the wide ones -- it triples the accumulator traffic,
            which is precisely what a 128x128 tile has no room for, so the
            cheap-looking 1.8x measured at ``BLOCK_M=32`` does not survive the
            blocking that makes this kernel fast.

            So tf32 it is, and 9e-04 is the honest cost of the tensor cores
            here. For scale: it is an order of magnitude under a bf16 ulp, and
            three orders under the quantisation error the codec is introducing
            on purpose. The mat-vec has no dot to place on a tensor core and so
            keeps 5e-07 for free; the two paths therefore do not agree to more
            than tf32, which is worth knowing when a result changes with batch
            size.

    Returns:
        Output ``(M, N)`` float32, summed over passes and K-chunks.
    """
    if not GEMM_TRELLIS_AVAILABLE:
        raise RuntimeError("triton is not available")
    check_window_table(table, trellis_bits)
    P, M, K = rx.shape
    N, words_per_row = words.shape[1], words.shape[2]
    vpw = 32 // bits
    wpg = group_size // vpw

    if not trellis_gemm_block_k(group_size, bits, 128):
        raise ValueError(f"no K-tile divides group_size {group_size} at "
                         f"{bits} bits")

    # A nominal tile, used only to size the K split. The split has to be decided
    # before the launch because it sizes the partial buffer and the grid's z
    # extent, while the tile the kernel actually runs is the autotuner's choice.
    # Splitting is what fills the device when the M and N tiles alone do not:
    # forcing it to 1 costs Clay 0.6 ms a forward, because `out` and `ff2` fall
    # to 112 tiles against 128 SMs.
    nominal_m = 32 if M <= 32 else (64 if M <= 128 else 128)
    nominal_n = 64
    tiles = triton.cdiv(M, nominal_m) * triton.cdiv(N, nominal_n)
    sms = _sm_count(rx.device.index or 0)
    max_chunks = max(1, words_per_row // wpg)
    split_k = min(max(1, (2 * sms) // max(1, tiles * P)), max_chunks)
    words_per_chunk = triton.cdiv(triton.cdiv(words_per_row, split_k), wpg) * wpg

    # The in-GEMM epilogue is only correct on a single band; with more, each
    # program holds a partial sum, so the epilogue moves to the fold instead.
    fused_epilogue = split_k * P == 1
    parts = torch.empty((split_k * P * M, N), dtype=torch.float32,
                        device=rx.device)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]),
                split_k * P)

    _gemm_trellis_kernel[grid](
        rx, words, scale, table, parts,
        bias if fused_epilogue and bias is not None else parts,
        M, N, K, words_per_chunk,
        rx.stride(0), rx.stride(1),
        words.stride(0), words.stride(1),
        scale.stride(0), scale.stride(1), scale.stride(2),
        parts.stride(0), parts.stride(1),
        GROUP_SIZE=group_size, WORDS_PER_GROUP=wpg,
        BITS=bits, L=trellis_bits, VPW=vpw, SPLIT_K=split_k, PREC=precision,
        HAS_BIAS=fused_epilogue and bias is not None,
        ACT_GELU=fused_epilogue and gelu,
    )
    if fused_epilogue:
        # Nothing to fold: the single band already is the answer, and summing a
        # length-one axis is a full-size copy of it. That copy is not free at
        # these shapes -- one per layer, 0.94 ms of a 15.8 ms Clay forward at
        # P=1, where split_k is 1 because the M and N tiles fill the device on
        # their own. Returning the buffer also drops the second allocation.
        return parts.view(M, N)
    # More than one band, so the epilogue moves into the fold rather than being
    # three more passes over the output after it. Bands are summed in index
    # order, so two identical calls return identical bits.
    return fold_partials(parts, split_k * P, M, N, bias, gelu)
