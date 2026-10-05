"""E8P codec: encode to 16-bit codes and decode back, exactly.

`experiments/e8p_stage.py` established that the padded E8 construction of QuIP\\#
reaches ``rho^2 = 0.0912`` as a data-free stage quantiser, but it returns
*reconstructed values*: it computes the winning codeword and throws the index
away. A shippable checkpoint needs the index. This module keeps it.

The 16-bit layout, which is exactly the 2^16 count the construction is built to
hit (``2 shifts x 256 sources x 128 valid sign patterns = 65536``):

    bits  0..7   j, an entry of the 256-word source codebook S = |D8h|
    bits  8..14  the signs of coordinates 0..6, one bit each (1 = negative)
    bit   15     the shift c: 0 -> +1/4, 1 -> -1/4

Coordinate 7's sign is *not* stored. It is recovered from the parity constraint
that puts the codeword in D8h: the number of negated coordinates must have
parity ``par[j]``, which pins the last sign given the other seven. That is the
whole reason the construction fits 8 dimensions into 16 bits at 2 bits per
coordinate, and it is why the resident table is 256 x 8 rather than 2^16 x 8:
1 KiB in fp16, against 1 MiB for the stored-table alternative.

Nothing here is fitted to data. S is a fixed lattice-derived set and the single
scale is tuned against the *known* spherical marginal, so the zero-data
constraint survives encoding.
"""

from __future__ import annotations

import itertools
import math
import os

import torch

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

# Coordinates per codeword. Fixed by the construction, not a tunable.
E8P_DIM = 8
# Entries in the source codebook.
E8P_SOURCE_SIZE = 256
# Codes per program in the fused decode. Each carries 8 coordinates, so this is
# 1024 output values, which is where the trellis kernel also flattened out.
_DECODE_BLOCK = 128


if HAVE_TRITON:

    @triton.jit
    def _decode_kernel(codes_ptr, S_ptr, par_ptr, out_ptr, n, scale,
                       DIM: tl.constexpr, BLOCK: tl.constexpr):
        """Uint16 codes straight to values, with nothing written in between.

        The torch path builds the sign pattern with a seven-element list
        comprehension and a stack, so a decode costs several full-size integer
        tensors before a single value is produced. Here the code stays in a
        register and the only DRAM traffic is two bytes in per eight values and
        four bytes out per value. Both lookup tables are 8 KB or smaller and
        stay in L1.
        """
        e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = e < n
        code = tl.load(codes_ptr + e, mask=live, other=0).to(tl.int32)

        j = code & 0xFF
        c = tl.arange(0, DIM)
        # Coordinates 0..6 carry an explicit sign bit; coordinate 7 is recovered
        # from the parity the source entry requires, which is the whole reason a
        # 2^16 codebook fits in 16 bits.
        raw = (code[:, None] >> (8 + c[None, :])) & 1
        explicit = c[None, :] < DIM - 1
        nsum = tl.sum(tl.where(explicit, raw, 0), axis=1)
        # `& 1` rather than `% 2`: the difference can be negative, and Triton's
        # remainder takes the sign of the dividend where torch's does not.
        last = (tl.load(par_ptr + j, mask=live, other=0) - nsum) & 1
        neg = tl.where(explicit, raw, last[:, None])

        sgn = tl.where(neg == 1, -1.0, 1.0)
        shift = tl.where((code >> 15) & 1 == 0, 0.25, -0.25)
        val = (sgn * tl.load(S_ptr + j[:, None] * DIM + c[None, :],
                             mask=live[:, None], other=0.0)
               + shift[:, None]) * scale
        tl.store(out_ptr + e[:, None] * DIM + c[None, :], val,
                 mask=live[:, None])


def d8h_abs_source(device: torch.device | str = "cpu") -> torch.Tensor:
    """The 256-entry source codebook ``S`` of E8P: absolute values from |D8h|.

    Coordinates of D8h lie in ``Z + 1/2``, so ``|D8h|`` entries lie in
    ``{0.5, 1.5, 2.5, 3.5}^8``. We take every pattern with ``norm^2 <= 10`` and
    pad with the shortest remaining ones (``norm^2 = 12``) up to 256, which is
    the 227 + 29 split of the original construction.

    Args:
        device: Device for the returned table.

    Returns:
        Float32 ``(256, 8)`` table of non-negative magnitudes.
    """
    vals = torch.tensor([0.5, 1.5, 2.5, 3.5])
    grids = torch.meshgrid(*([vals] * E8P_DIM), indexing="ij")
    pts = torch.stack([g.reshape(-1) for g in grids], dim=1)
    nrm = pts.pow(2).sum(1)
    keep = pts[nrm <= 10.0 + 1e-6]
    pad = pts[(nrm > 10.0 + 1e-6) & (nrm <= 12.0 + 1e-6)]
    pad = pad[pad.pow(2).sum(1).argsort()]
    S = torch.cat([keep, pad], dim=0)[:E8P_SOURCE_SIZE]
    assert S.shape == (E8P_SOURCE_SIZE, E8P_DIM), S.shape
    return S.to(device)


def parity_of(S: torch.Tensor) -> torch.Tensor:
    """Flip parity required for each magnitude pattern to land in D8h.

    The all-positive vector ``s`` has coordinate sum ``sum(s)``; D8h needs an
    even sum. Flipping coordinate ``i`` changes the sum by ``-2 s_i``, and
    ``2 s_i`` is an odd integer, so every flip flips the sum's parity. Hence the
    required number of flips has parity ``sum(s) mod 2``.

    Args:
        S: The ``(256, 8)`` source codebook.

    Returns:
        Int64 ``(256,)`` tensor of required flip parities, each 0 or 1.
    """
    return S.sum(1).round().long() % 2


def build_full_codebook(S: torch.Tensor) -> torch.Tensor:
    """Materialise all 2^16 codewords. Verification only; never used to decode.

    Args:
        S: The ``(256, 8)`` source codebook.

    Returns:
        Float32 ``(65536, 8)`` tensor of every codeword.
    """
    par = parity_of(S)
    signs = torch.tensor(list(itertools.product([1.0, -1.0], repeat=E8P_DIM)),
                         device=S.device)
    nflip = (signs < 0).sum(1)
    out = []
    for shift in (0.25, -0.25):
        for j in range(E8P_SOURCE_SIZE):
            ok = (nflip % 2) == par[j]
            out.append(S[j].unsqueeze(0) * signs[ok] + shift)
    cb = torch.cat(out, dim=0)
    assert cb.shape[0] == 1 << 16, cb.shape
    return cb.contiguous()


class E8PCodec:
    """Nearest-E8P-codeword encoder/decoder with one globally tuned scale.

    The scale is fitted by golden-section search against samples from the known
    spherical marginal, never against model weights, so the codec is data-free
    in the sense the paper claims throughout.
    """

    def __init__(self, d: int, n_probe: int = 20000, seed: int = 0,
                 chunk: int = 16384, device: torch.device | str = "cpu") -> None:
        """
        Args:
            d: Group size the codec will be applied within (sets the probe scale).
            n_probe: Number of probe coordinates for the scale search.
            seed: Seed for the probe sample.
            chunk: Blocks of 8 coordinates encoded per inner iteration.
            device: Device to build tables on.
        """
        self.device = torch.device(device)
        self.S = d8h_abs_source(self.device)
        self.par = parity_of(self.S)
        self.Ssq = self.S.pow(2).sum(1)
        # Kernel-side views of the two lookup tables, cast once. Together they
        # are 9 KB, which is why the fused decode never leaves L1 for them.
        self._S_f32 = self.S.to(torch.float32).contiguous()
        self._par_i32 = self.par.to(torch.int32).contiguous()
        self.chunk = chunk
        self.d = d

        g = torch.Generator(device=self.device).manual_seed(seed)
        probe = _sample_sphere(n_probe // (d // E8P_DIM) + 1, d, g,
                               self.device).reshape(-1, E8P_DIM)
        lo, hi = 1e-3 / math.sqrt(d), 4.0 / math.sqrt(d)
        for _ in range(28):
            m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
            d1 = (probe - self._decode_blocks(*self._encode_blocks(probe, m1), m1)).pow(2).sum().item()
            d2 = (probe - self._decode_blocks(*self._encode_blocks(probe, m2), m2)).pow(2).sum().item()
            hi, lo = (m2, lo) if d1 < d2 else (hi, m1)
        self.scale = 0.5 * (lo + hi)

    # ---------------------------------------------------------------- internals
    def _encode_blocks(self, blocks: torch.Tensor, scale: float):
        """Encode ``(m, 8)`` blocks at ``scale`` into (j, sgn, cbit).

        Returns the winning source index, the full 8-coordinate sign pattern
        (+1/-1), and the shift bit. Splitting encode from decode this way is
        what lets the caller keep the index instead of only the value.
        """
        x = blocks / scale
        best_d = best_j = best_sgn = best_c = None
        for cbit, c in enumerate((0.25, -0.25)):
            y = x - c
            a = y.abs()
            # ||a - s||^2 = ||a||^2 - 2 a.s + ||s||^2. ||a||^2 is constant within
            # a shift but NOT across the two shifts, so it must be kept: dropping
            # it makes the c = +1/4 vs c = -1/4 comparison meaningless.
            base = (a.pow(2).sum(1, keepdim=True)
                    + self.Ssq.unsqueeze(0) - 2.0 * (a @ self.S.T))
            # cheapest single flip for each (x, s), used when parity is violated
            pen = 4.0 * (a.unsqueeze(2) * self.S.T.unsqueeze(0)).min(dim=1).values
            nneg = (y < 0).sum(1)
            mism = ((nneg.unsqueeze(1) % 2) != self.par.unsqueeze(0))
            dist = base + torch.where(mism, pen, torch.zeros_like(pen))
            dmin, j = dist.min(dim=1)

            sgn = torch.where(y < 0, -1.0, 1.0)
            need = mism.gather(1, j.unsqueeze(1)).squeeze(1)
            k = (a * self.S[j]).argmin(dim=1)
            flip = torch.zeros_like(sgn, dtype=torch.bool)
            flip[torch.arange(x.shape[0], device=x.device), k] = True
            sgn = torch.where(flip & need.unsqueeze(1), -sgn, sgn)

            cb = torch.full_like(j, cbit)
            if best_d is None:
                best_d, best_j, best_sgn, best_c = dmin, j, sgn, cb
            else:
                take = dmin < best_d
                best_d = torch.where(take, dmin, best_d)
                best_j = torch.where(take, j, best_j)
                best_c = torch.where(take, cb, best_c)
                best_sgn = torch.where(take.unsqueeze(1), sgn, best_sgn)
        return best_j, best_sgn, best_c

    def _decode_blocks(self, j: torch.Tensor, sgn: torch.Tensor,
                       cbit: torch.Tensor, scale: float) -> torch.Tensor:
        """Rebuild ``(m, 8)`` values from (j, sgn, cbit) at ``scale``."""
        c = torch.where(cbit == 0, 0.25, -0.25).unsqueeze(1)
        return (sgn * self.S[j] + c) * scale

    # ------------------------------------------------------------------- public
    def encode(self, u: torch.Tensor) -> torch.Tensor:
        """Encode a tensor whose last-axis length is a multiple of 8.

        Args:
            u: Values to quantise.

        Returns:
            Uint16 codes of shape ``u.shape[:-1] + (u.shape[-1] // 8,)``.
        """
        flat = u.reshape(-1, E8P_DIM)
        out = torch.empty(flat.shape[0], dtype=torch.int32, device=flat.device)
        for i in range(0, flat.shape[0], self.chunk):
            j, sgn, cbit = self._encode_blocks(flat[i:i + self.chunk], self.scale)
            neg = (sgn < 0).to(torch.int32)
            # Only coordinates 0..6 are stored; coordinate 7 rides the parity.
            bits = torch.zeros_like(j, dtype=torch.int32)
            for b in range(E8P_DIM - 1):
                bits |= neg[:, b] << b
            out[i:i + self.chunk] = (j.to(torch.int32)
                                     | (bits << 8)
                                     | (cbit.to(torch.int32) << 15))
        return out.to(torch.uint16).reshape(*u.shape[:-1], u.shape[-1] // E8P_DIM)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode uint16 codes back to values.

        Args:
            codes: Uint16 codes as produced by :meth:`encode`.

        Returns:
            Float32 values of shape ``codes.shape[:-1] + (codes.shape[-1] * 8,)``.
        """
        flat = codes.reshape(-1).to(torch.int32)
        if HAVE_TRITON and flat.is_cuda:
            n = flat.numel()
            vals = torch.empty((n, E8P_DIM), dtype=torch.float32,
                               device=flat.device)
            _decode_kernel[(triton.cdiv(n, _DECODE_BLOCK),)](
                flat.contiguous(), self._S_f32, self._par_i32, vals, n,
                self.scale, DIM=E8P_DIM, BLOCK=_DECODE_BLOCK, num_warps=4)
        else:
            j = flat & 0xFF
            cbit = (flat >> 15) & 1
            neg = torch.stack([(flat >> (8 + b)) & 1
                               for b in range(E8P_DIM - 1)], dim=1)
            # Recover coordinate 7: total negated count must have parity par[j].
            last = (self.par[j] - neg.sum(1)) % 2
            neg = torch.cat([neg, last.unsqueeze(1)], dim=1)
            sgn = torch.where(neg == 1, -1.0, 1.0)
            vals = self._decode_blocks(j, sgn, cbit, self.scale)
        return vals.reshape(*codes.shape[:-1], codes.shape[-1] * E8P_DIM)

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        """Quantise in one step, returning reconstructed values."""
        return self.decode(self.encode(u)).reshape(u.shape)


def _sample_sphere(n: int, d: int, gen: torch.Generator,
                   device: torch.device) -> torch.Tensor:
    x = torch.randn(n, d, generator=gen, device=device)
    return x / x.norm(dim=1, keepdim=True)
