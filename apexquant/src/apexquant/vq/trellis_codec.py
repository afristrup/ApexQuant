"""Trellis codec: the bitshift TCQ stage, keeping the codes.

`experiments/vector_stage_rho.make_tcq_stage` measures this stage's ``rho`` and
`progressive_vector_stage_llm` runs whole ladders with it, but both only ever
need *reconstructed values*: they run Viterbi, emit the winning sequence and
throw the path away. A shippable checkpoint needs the path. This module keeps it,
behind the same ``encode``/``decode`` interface as
:class:`~apexquant.vq.scalar_codec.ScalarCodec` and
:class:`~apexquant.vq.e8p_codec.E8PCodec`, so the progressive recursion and
the converter take it without knowing anything new.

The construction is QTIP's, unchanged. State is an ``L``-bit shift register; each
step shifts in ``k`` fresh bits, so ``s' = ((s << k) | b) & (2^L - 1)``. The
codeword attached to a state is a *computed* pseudo-Gaussian hash of the state
rather than a stored table, which is what makes the stage data-free and what makes
its decode table-free. Encoding is exact Viterbi over the ``d``-step trellis.

Recovering the code is easier than it looks. The transition rule puts the input
symbol in the low ``k`` bits of the successor, so ``b_t = state_t & (2^k - 1)``:
the backtrack already walks the winning states, and reading their low bits gives
the symbol sequence directly. Decoding then needs no search at all, only the same
shift-register walk forward from the initial state, which is why this stage is
cheap to decode and expensive to encode.

Cost is worth stating plainly. Each step scores ``2^L`` states against ``2^k``
predecessors, so encoding is some three orders of magnitude dearer than the scalar
or E8P stages, and the survivor table is the memory bound. Decoding is a ``d``-step
walk with no table, so the asymmetry runs the right way for a checkpoint that is
written once and read many times. :meth:`TrellisCodec._viterbi_codes` cuts both by
``2^k`` by sharing the ACS minimum across the states that share a predecessor set,
and ``beam`` trades exactness for a further ``2^L / B``.

The constant in front of that cost was set by where the path metric lived, not by
the arithmetic. Written in torch the ``2^L`` costs are a tensor, so every one of
the ``d`` steps reads and rewrites them through DRAM: at ``L=12, d=512`` that is
33 MB of traffic per row against 0.5 MB of survivors, and the encoder ran at
memory bandwidth doing almost no work. :meth:`TrellisCodec._acs_codes` keeps the
whole cost vector in registers and loops the ``d`` steps inside one launch, which
leaves the survivor table as the only thing crossing DRAM. That is worth 44x at
the shipped ``L=12`` and 68x at ``L=14``, the factor rising with ``L`` because the
traffic it removes does and the survivors it keeps do not. Both are on an
otherwise idle card; a contended one reads about half that, which is why
`experiments/trellis_encoder_levers.py` takes a best-of-N.
"""

from __future__ import annotations

import math
import os

import torch

from .scalar_codec import pack_indices, unpack_indices

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                  # CPU-only or no triton
    HAVE_TRITON = False

# Escape hatch, and the only supported way to compare the two paths on one
# machine. The fallback is the reference implementation the kernel is checked
# against, so this also answers "is the kernel the thing that changed this
# number" without editing code.
if os.environ.get("APEXQUANT_NO_TRITON") == "1":
    HAVE_TRITON = False

DEFAULT_TRELLIS_BITS = 12
# Flat over output elements, so this is just "enough work per program". Measured
# on a 4090 at 59M values: 256, 512, 1024 and 2048 all land within noise of
# 0.30 ms, which is the bandwidth the write itself costs, so the choice inside
# that range does not matter.
_WALK_BLOCK = 1024

# Above this the cost vector stops fitting in registers and the kernel spills,
# which costs more than the DRAM round trip it exists to avoid. The kernel holds
# the costs, the codewords and their squares, so 3 * 2^L floats over at most 1024
# threads: 48 a thread at 2^12 and again at 2^14, since the block widens with the
# trellis, but 192 at 2^16, where widening has run out. So 2^14 is the last width
# that fits, and `experiments/trellis_encoder_levers.py` measures what that is
# worth -- 68x at L=14 against 2.5x at L=16, the latter already spilling.
# `_acs_codes` therefore covers the shipped config and the crossover regime
# around it, and `_viterbi_codes` remains the general one.
_ACS_MAX_STATES = 1 << 14
# Rows per traceback program. The walk is one dependent byte load per step, so
# what hides the latency is having many rows in flight inside a program, not
# many programs; 128 fills the pipe and the stage is ~4% of the encode either way.
_TRACEBACK_BLOCK = 128


if HAVE_TRITON:

    @triton.jit
    def _walk_kernel(codes_ptr, cw_ptr, out_ptr, total, dd, code_stride, scale,
                     BITS: tl.constexpr, PER: tl.constexpr, M: tl.constexpr,
                     BLOCK: tl.constexpr):
        """Packed symbols to codewords, state kept in registers.

        Flattened over ``(row, position)`` rather than gridded one program per
        row. A row is only ``dd`` values, so a per-row grid launches hundreds of
        thousands of programs that each write a kilobyte, and the launch
        accounting rather than the memory system sets the pace: measured 2.50 ms
        per-row against 0.30 ms flattened, for identical output.

        Nothing intermediate reaches DRAM. The ``M`` neighbour loads all fall
        inside one row's few dozen packed bytes, and the codeword table is a few
        KB, so both stay in L1: the traffic is the codes in and the values out.
        """
        e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = e < total
        row, pos = e // dd, e % dd
        base = codes_ptr + row.to(tl.int64) * code_stride

        # Symbol at ``pos - i`` occupies bit field ``i`` of the state; positions
        # before the start of a row contributed nothing, which is exactly the
        # zero the register was initialised with.
        state = tl.zeros((BLOCK,), dtype=tl.int32)
        for i in tl.static_range(M):
            p = pos - i
            # Clamped for the address and the shift; the mask is what actually
            # zeroes the out-of-range lanes. A negative shift count would be
            # undefined even on a lane whose result is later discarded.
            pc = tl.maximum(p, 0)
            byte = tl.load(base + (pc // PER), mask=live & (p >= 0), other=0)
            sym = (byte.to(tl.int32) >> ((pc % PER) * BITS)) & ((1 << BITS) - 1)
            state |= sym << (BITS * i)

        # Scale folded in. Applied outside it would be a second full pass over
        # the output, reading and writing every value again to multiply by a
        # constant, which measured 0.82 ms against 0.31 ms for the kernel alone.
        tl.store(out_ptr + e,
                 tl.load(cw_ptr + state, mask=live, other=0.0) * scale,
                 mask=live)


    @triton.jit
    def _acs_kernel(x_ptr, cw_ptr, cwsq_ptr, back_ptr, final_ptr, dd,
                    K: tl.constexpr, M: tl.constexpr, S: tl.constexpr):
        """One row's whole forward pass, cost vector never leaving registers.

        The torch loop this replaces is the same arithmetic; what changed is
        where ``cost`` lives. Held as a tensor it is read and rewritten four
        times a step through DRAM, and at ``L=12, d=512`` that is 33 MB a row
        against the 0.5 MB of survivors the algorithm actually has to record --
        so the encoder ran at memory bandwidth while the ALUs idled. Held in
        registers the traffic is the survivors and nothing else, and one launch
        covers all ``d`` steps rather than ``d`` launches covering one each.

        A program is one row and owns the register file for it, which is why the
        grid is ``n`` and not a tiling: rows are independent, steps are not.

        Args:
            x_ptr: ``(n, dd)`` float32, contiguous, in codeword units.
            cw_ptr, cwsq_ptr: ``(S,)`` codewords and their squares.
            back_ptr: ``(n, dd, M)`` uint8 survivor table, written here.
            final_ptr: ``(n,)`` int32 winning end state, written here.
        """
        row = tl.program_id(0).to(tl.int64)
        s = tl.arange(0, S)
        cw = tl.load(cw_ptr + s)
        cwsq = tl.load(cwsq_ptr + s)
        # The walk starts at state 0; everything else is unreachable at t=0.
        cost = tl.where(s == 0, 0.0, 1e30)
        xb = x_ptr + row * dd
        bb = back_ptr + row * dd * M + tl.arange(0, M)
        for t in range(dd):
            # cost[j*M + m] is the j-th predecessor of every state with
            # s >> k == m, so (K, M) makes the survivor a min over the leading
            # axis -- the same merge-class reduction as the torch path, and the
            # same reason no gather is needed.
            cv = tl.reshape(cost, (K, M))
            tl.store(bb + t * M, tl.argmin(cv, axis=0).to(tl.uint8))
            # Successors of class m are the K states m*K + b, so writing the new
            # costs in (M, K) order leaves the state layout unchanged. The
            # constant u^2 of (u - c)^2 is dropped: it cannot move an argmin.
            u = tl.load(xb + t)
            best = tl.min(cv, axis=0)
            cost = tl.reshape(tl.broadcast_to(best[:, None], (M, K)), (S,))
            cost = cost + cwsq - 2.0 * u * cw
        tl.store(final_ptr + row, tl.argmin(cost, axis=0))

    @triton.jit
    def _traceback_kernel(back_ptr, final_ptr, codes_ptr, n, dd,
                          K: tl.constexpr, M: tl.constexpr, BITS: tl.constexpr,
                          BLOCK: tl.constexpr):
        """Backtrack ``BLOCK`` rows at once, all ``dd`` steps in one launch.

        Vectorised over rows for the same reason the forward pass is not: the
        walk is strictly sequential in ``t`` but rows share nothing, so a program
        carries ``BLOCK`` independent scalar states and issues ``BLOCK``
        independent loads a step. In torch this was ``dd`` launches of a gather
        that touched one byte a row, which at 512 steps a chunk is pure launch
        accounting.
        """
        r = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        live = r < n
        state = tl.load(final_ptr + r, mask=live, other=0)
        for t in range(dd - 1, -1, -1):
            # The transition puts the input symbol in the low k bits of the
            # successor, so the symbol that produced this state is its low bits.
            tl.store(codes_ptr + r * dd + t, (state & (K - 1)).to(tl.uint8),
                     mask=live)
            m = state >> BITS
            j = tl.load(back_ptr + r * dd * M + t * M + m, mask=live, other=0)
            state = j.to(tl.int32) * M + m         # == (m | (j << (L - k)))


def _generalized_gaussian_quantile(u: torch.Tensor, shape: float,
                                   grid: int = 1 << 16,
                                   span: float = 12.0) -> torch.Tensor:
    """Inverse CDF of a unit-variance generalized Gaussian, by grid inversion.

    There is no closed form and torch has no inverse incomplete gamma, so the
    CDF is built on a dense grid and inverted by search. Two properties are
    load-bearing rather than incidental. It is deterministic, so the same shape
    yields the same table on every run. And it is computed entirely on the CPU in
    float64 before being moved, so it is device-independent: the artifact records
    only the shape, and a decoder that rebuilt the table on a different device
    class and got a slightly different answer would rescale every weight with
    nothing raising.
    """
    alpha = math.sqrt(math.exp(math.lgamma(1.0 / shape) - math.lgamma(3.0 / shape)))
    x = torch.linspace(-span, span, grid, dtype=torch.float64)
    log_w = -((x / alpha).abs() ** shape)
    w = torch.exp(log_w - log_w.max())
    cdf = torch.cumsum(w, dim=0)
    cdf = cdf / cdf[-1]
    idx = torch.searchsorted(cdf, u.detach().to("cpu", torch.float64).contiguous())
    return x[idx.clamp_(max=grid - 1)]


def _codeword_table(trellis_bits: int, seed: int,
                    device: torch.device, shape: float = 2.0) -> torch.Tensor:
    """Computed pseudo-Gaussian codeword per state. No table is stored on disk.

    A cheap integer hash of the state index, mapped through an inverse CDF and
    normalised to unit variance. At the default ``shape=2.0`` that CDF is the
    normal one and this is QTIP's 1MAD construction, identical to
    ``vector_stage_rho.make_tcq_stage`` so the two agree on every codeword.

    ``shape`` is the 1MAD-``p`` axis. It replaces the target marginal with a
    generalized Gaussian of that exponent, leaving the trellis, the cosets and
    the search untouched: only the values the states carry move. ``2.0`` is
    Gaussian, ``1.0`` Laplace, and below ``1`` the hyper-Laplacian HQQ uses to
    model outlier error. The point is not source matching, since the source is
    already close to Gaussian at deployed fan-in, but error-tail control, so
    expect ``rho`` to rise and the tail to shorten.

    ``shape=2.0`` takes the exact ``erfinv`` path rather than the grid, so it
    reproduces every previously published artifact bit for bit. Any other value
    is a different codebook and a different artifact.
    """
    S = 1 << trellis_bits
    s_idx = torch.arange(S, device=device, dtype=torch.int64)
    h = (s_idx * 2654435761 + 0x9E3779B9 + seed * 1013904223) & 0xFFFFFFFF
    h = (h ^ (h >> 15)) * 2246822519 & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 3266489917 & 0xFFFFFFFF
    h = h ^ (h >> 16)
    unif = (h.to(torch.float64) + 0.5) / 2.0 ** 32
    if shape == 2.0:
        cw = (torch.erfinv(2.0 * unif - 1.0) * math.sqrt(2.0)).to(torch.float32)
    else:
        if shape <= 0.0:
            raise ValueError(f"shape must be > 0, got {shape}")
        cw = _generalized_gaussian_quantile(unif, shape).to(device, torch.float32)
    return cw / cw.std()


class TrellisCodec:
    """Bitshift-trellis quantiser that emits its path.

    Attributes:
        bits: Rate in bits per coordinate, and the width of one stored symbol.
        trellis_bits: Shift-register width ``L``; ``2**L`` states.
        d: Group size, which sets the per-coordinate scale of a unit-norm group.
        beam: Survivors kept per step. Zero is exact Viterbi over all ``2**L``
            states; a positive value runs the M-algorithm instead, which is
            cheaper by ``2**L / beam`` and no longer optimal.
        shape: Generalized-Gaussian exponent of the codeword marginal, the
            1MAD-``p`` axis. ``2.0`` is QTIP's Gaussian target and is the only
            value that reproduces previously published artifacts. Anything else
            must be recorded alongside the artifact, since a decoder that
            rebuilds the table at the wrong shape decodes every weight wrong.
    """

    def __init__(self, d: int, bits: int = 2,
                 trellis_bits: int = DEFAULT_TRELLIS_BITS, seed: int = 0,
                 device: torch.device | str = "cpu", beam: int = 0,
                 shape: float = 2.0) -> None:
        if 8 % bits:
            raise ValueError(f"bits must divide 8 for dense packing, got {bits}")
        if trellis_bits % bits:
            raise ValueError(
                f"trellis_bits {trellis_bits} must be a multiple of bits {bits}; "
                "the shift register consumes whole symbols")
        if beam and not 0 < beam <= 256:
            raise ValueError(f"beam must be in [1, 256], got {beam}; the parent "
                             "slot is stored in a byte")
        self.device = torch.device(device)
        self.beam = beam
        self.bits = bits
        self.trellis_bits = trellis_bits
        self.seed = seed
        self.shape = float(shape)
        self.d = d
        self.S = 1 << trellis_bits
        self.K = 1 << bits
        self.codeword = _codeword_table(trellis_bits, seed, self.device, self.shape)
        # The kernel indexes this directly, so cast once here rather than on
        # every decode: the table is a few KB and a decode is called per pass
        # per tensor per forward.
        self._codeword_f32 = self.codeword.to(torch.float32)
        # Coordinates of a unit-norm d-vector are ~N(0, 1/d); the codewords are
        # unit variance, so the group is scaled into the codeword's units and back.
        self.scale = 1.0 / math.sqrt(d)

        # Number of merge classes: the predecessors of state s are
        # {(s >> k) | (j << (L - k))} for j in [0, 2^k), which depend on s only
        # through m = s >> k. So the 2^k states sharing an m share one ACS
        # minimum, and there are M = 2^(L-k) of them per step rather than 2^L.
        self.M = self.S >> bits
        # Branch metrics are rank two: (u - c)^2 = u^2 - 2 u c + c^2, and u^2 is
        # constant across states, so it drops out of the argmin. Keeping c^2
        # here turns the per-step metric into one AXPY over the state axis.
        self._codeword_sq = self.codeword.pow(2)

    # ---------------------------------------------------------------- internals
    def _rows_per_chunk(self, dd: int) -> int:
        """Backpointers dominate memory; cap the batch so they fit in 2 GiB.

        Exact Viterbi stores one survivor decision per merge class per step, so
        ``n * dd * M`` bytes. Beam search stores a parent slot and a symbol per
        beam entry per step, so ``2 * n * dd * B``.
        """
        per_row = dd * (2 * self.beam if self.beam else self.M)
        return max(1, (1 << 31) // max(1, per_row))

    def _viterbi_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Exact Viterbi; returns the winning symbol sequence, not the values.

        Two structural facts do the work here, and both are properties of the
        bitshift trellis rather than of Viterbi in general.

        The first is that the ACS minimum is shared. Predecessors of state ``s``
        are ``{(s >> k) | (j << (L-k))}``, which depend on ``s`` only through
        ``m = s >> k``, so the ``2^k`` states sharing an ``m`` -- the de Bruijn
        butterfly -- have the same predecessor set and therefore the same
        minimum. Taking it once per ``m`` rather than once per state costs
        ``2^L`` comparisons a step instead of ``2^(L+k)``, and needs no gather at
        all: laying ``cost`` out as ``(K, M)`` makes ``j`` the leading axis and
        the reduction a plain ``min`` over it. The survivor table shrinks by the
        same factor, which is what sets ``_rows_per_chunk``.

        The second is that the branch metric is rank two. ``(u_t - c)^2`` expands
        to ``u_t^2 - 2 u_t c + c^2``, so with ``c`` and ``c^2`` precomputed a
        step is one AXPY against the state axis rather than a subtract and a
        square. The ``u_t^2`` term is constant across states and cannot move the
        argmin, but it is kept and added to the ``2^(L-k)`` survivors rather than
        to the ``2^L`` new costs, which is free. Dropping it is *not* free: an
        input that lies exactly on a codeword then scores ``-c^2`` a step instead
        of exactly zero, so the true path's metric accumulates rounding where it
        should accumulate nothing, and at ``L=16`` that is enough to lose it to a
        path ``1e-12`` per coordinate worse. ``serving_evals/verify.py`` asserts
        ``encode(decode(c)) == c``, which is exactly that input.

        Args:
            x: ``(n, dd)`` in codeword units, i.e. already divided by ``scale``.

        Returns:
            Int64 ``(n, dd)`` symbols in ``[0, 2**bits)``.
        """
        n, dd = x.shape
        k, S, K, M = self.bits, self.S, self.K, self.M
        cost = torch.full((n, S), 1e30, device=x.device)
        cost[:, 0] = 0.0                                   # the walk starts at 0
        back = torch.empty((n, dd, M), dtype=torch.uint8, device=x.device)
        c, c2 = self.codeword.view(1, S), self._codeword_sq.view(1, M, K)
        for t in range(dd):
            # cost[:, j*M + m] is the j-th predecessor of every state with
            # s >> k == m, so the survivor per merge class is a min over j.
            best, arg = cost.view(n, K, M).min(dim=1)                  # (n, M)
            back[:, t] = arg.to(torch.uint8)
            # Successors of class m are the K states m*K + b, so the (M, K)
            # layout of the new costs is the state layout unchanged. u_t^2 goes
            # in here, on the survivors, rather than on the K times wider result.
            best += x[:, t].square().unsqueeze(1)
            cost = (best.unsqueeze(2) + c2).view(n, S)
            cost.addcmul_(x[:, t].unsqueeze(1), c, value=-2.0)

        state = cost.argmin(dim=1)
        codes = torch.empty((n, dd), dtype=torch.int64, device=x.device)
        for t in range(dd - 1, -1, -1):
            # The transition puts the input symbol in the low k bits of the
            # successor, so the symbol that produced this state is simply its
            # low k bits. That is the whole trick that makes the path storable.
            codes[:, t] = state & (K - 1)
            m = state >> k
            j = back[:, t].gather(1, m.unsqueeze(1)).squeeze(1).to(torch.int64)
            state = j * M + m                    # == (m | (j << (L - k)))
        return codes

    def _acs_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Exact Viterbi in two launches. Same algorithm as :meth:`_viterbi_codes`.

        Not the same *output*, quite. Ties in the ACS minimum break to whichever
        index the reduction happens to keep, and torch and triton do not agree on
        that, so about one symbol in 3000 differs at ``L=12``. Both are winning
        paths of equal metric, which is the sense in which they are equally good
        encodings: measured ``rho^2`` agrees to every digit printed. Read
        agreement of the codes as a diagnostic and never as an error measure.

        Args:
            x: ``(n, dd)`` in codeword units, i.e. already divided by ``scale``.

        Returns:
            Uint8 ``(n, dd)`` symbols in ``[0, 2**bits)``. The torch path returns
            int64; both feed ``pack_indices`` unchanged.
        """
        n, dd = x.shape
        x = x.contiguous()
        back = torch.empty((n, dd, self.M), dtype=torch.uint8, device=x.device)
        final = torch.empty(n, dtype=torch.int32, device=x.device)
        codes = torch.empty((n, dd), dtype=torch.uint8, device=x.device)
        _acs_kernel[(n,)](
            x, self.codeword, self._codeword_sq, back, final, dd,
            K=self.K, M=self.M, S=self.S,
            num_warps=max(4, min(32, self.S // 512)), num_stages=1)
        _traceback_kernel[(triton.cdiv(n, _TRACEBACK_BLOCK),)](
            back, final, codes, n, dd,
            K=self.K, M=self.M, BITS=self.bits, BLOCK=_TRACEBACK_BLOCK,
            num_warps=4)
        return codes

    def _beam_codes(self, x: torch.Tensor) -> torch.Tensor:
        """The M-algorithm: carry the best ``beam`` states, not all ``2^L``.

        The trade is only attractive because of what a pruning error costs here.
        In error correction, dropping the true path is a frame error you can
        only bound probabilistically. In quantisation every surviving path is a
        valid encoding, so a pruning error buys a slightly worse reconstruction
        and nothing else -- there is no correctness obligation, only a rho to
        measure. ``experiments/trellis_encoder_levers.py`` measures it.

        The merge step is exact and worth keeping. Two beam entries whose low
        ``L-k`` bits agree have identical successor states and therefore
        identical futures, so the dearer one can be dropped with no loss at all;
        without it the beam fills with entries that are already the same state
        by another name, and at ``B`` comparable to ``2^(L-k)`` that is most of
        them.

        Args:
            x: ``(n, dd)`` in codeword units.

        Returns:
            Int64 ``(n, dd)`` symbols in ``[0, 2**bits)``.
        """
        n, dd = x.shape
        k, S, K, B = self.bits, self.S, self.K, self.beam
        c, c2 = self.codeword, self._codeword_sq
        b_idx = torch.arange(K, device=x.device).view(1, 1, K)
        # The walk starts at state 0, so the beam is one wide and grows K-fold a
        # step until it saturates.
        state = torch.zeros((n, 1), dtype=torch.int64, device=x.device)
        cost = torch.zeros((n, 1), device=x.device)
        par = torch.zeros((n, dd, B), dtype=torch.uint8, device=x.device)
        sym = torch.zeros((n, dd, B), dtype=torch.uint8, device=x.device)
        for t in range(dd):
            succ = (((state.unsqueeze(2) << k) | b_idx) & (S - 1)).reshape(n, -1)
            base = (cost + x[:, t].square().unsqueeze(1)).unsqueeze(2)
            cand = (base.expand(-1, -1, K).reshape(n, -1)
                    + c2[succ]).addcmul_(x[:, t].unsqueeze(1), c[succ],
                                         value=-2.0)
            w = min(B, cand.shape[1])
            cost, sel = cand.topk(w, dim=1, largest=False)             # sorted
            state = succ.gather(1, sel)
            par[:, t, :w] = torch.div(sel, K, rounding_mode="floor").to(torch.uint8)
            sym[:, t, :w] = (sel % K).to(torch.uint8)
            if w == B:
                # Duplicate successors are already adjacent in cost order, so
                # first-occurrence keeps the cheaper of each merged pair.
                order = state.argsort(dim=1, stable=True)
                srt = state.gather(1, order)
                first = torch.ones_like(srt, dtype=torch.bool)
                first[:, 1:] = srt[:, 1:] != srt[:, :-1]
                cost = cost.masked_fill(
                    ~torch.zeros_like(first).scatter_(1, order, first), 1e30)

        slot = cost.argmin(dim=1)
        codes = torch.empty((n, dd), dtype=torch.int64, device=x.device)
        for t in range(dd - 1, -1, -1):
            codes[:, t] = sym[:, t].gather(1, slot.unsqueeze(1)).squeeze(1)
            slot = par[:, t].gather(1, slot.unsqueeze(1)).squeeze(1).to(torch.int64)
        return codes

    def _walk(self, codes: torch.Tensor) -> torch.Tensor:
        """Forward shift-register walk: symbols to codewords. No search.

        The reference implementation, and the fallback when the fused kernel
        cannot run. Reads as the transition rule states it, one step at a time,
        which is what makes it worth keeping even though the kernel is what
        actually runs: every fast path in this file is checked against it.
        """
        n, dd = codes.shape
        state = torch.zeros(n, dtype=torch.int64, device=codes.device)
        out = torch.empty((n, dd), dtype=torch.float32, device=codes.device)
        for t in range(dd):
            state = ((state << self.bits) | codes[:, t]) & (self.S - 1)
            out[:, t] = self.codeword[state]
        return out

    def _walk_packed(self, packed: torch.Tensor, dd: int) -> torch.Tensor:
        """Packed codes straight to scaled values, unpack and scale fused in.

        Note the difference from :meth:`_walk`, which returns bare codewords:
        this applies ``self.scale`` too, so the caller must not apply it again.

        The walk is written as a scan but it is not one. The register is ``L``
        bits wide and eats ``k`` bits per step, and the constructor requires
        ``k`` to divide ``L``, so after ``m = L / k`` steps nothing of the
        earlier state survives: the state at position ``t`` is exactly the last
        ``m`` symbols packed, zero-padded before the sequence starts. Position
        ``t`` therefore depends on a fixed window, not on all of its history.

        That observation alone does not buy anything in torch, where writing it
        as ``m`` vector ops means materialising a 64-bit state for every
        coordinate -- eight bytes per element to avoid a loop that only ever
        held eight bytes per *row*, which measured slower than the scan it
        replaced. The window only pays once the state stays in registers, which
        is what the kernel is for. DRAM traffic is then the codes in and the
        values out, and nothing else.
        """
        n = packed.shape[0]
        out = torch.empty((n, dd), dtype=torch.float32, device=packed.device)
        total = n * dd
        _walk_kernel[(triton.cdiv(total, _WALK_BLOCK),)](
            packed, self._codeword_f32, out, total, dd, packed.shape[1],
            self.scale, BITS=self.bits, PER=8 // self.bits,
            M=self.trellis_bits // self.bits,
            BLOCK=_WALK_BLOCK, num_warps=4)
        return out

    # ------------------------------------------------------------------- public
    def encode(self, u: torch.Tensor) -> torch.Tensor:
        """Encode ``(n, gs)`` values to densely packed uint8 codes."""
        n, dd = u.shape
        bs = self._rows_per_chunk(dd)
        if self.beam:
            search = self._beam_codes
        elif (HAVE_TRITON and u.is_cuda and self.S <= _ACS_MAX_STATES):
            search = self._acs_codes
        else:
            search = self._viterbi_codes
        parts = []
        for i in range(0, n, bs):
            sym = search(u[i:i + bs] / self.scale)
            parts.append(pack_indices(sym.to(torch.uint8), self.bits))
        return torch.cat(parts, dim=0)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode packed uint8 codes back to values."""
        flat = codes.reshape(-1, codes.shape[-1])
        dd = flat.shape[-1] * (8 // self.bits)
        if HAVE_TRITON and flat.is_cuda:
            vals = self._walk_packed(flat.contiguous(), dd)   # scale folded in
        else:
            sym = unpack_indices(flat, self.bits).to(torch.int64)
            vals = self._walk(sym) * self.scale
        return vals.reshape(*codes.shape[:-1], dd)

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(u)).reshape(u.shape)
