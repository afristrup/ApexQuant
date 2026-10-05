"""QuIP#'s E8P codebook as a data-free stage quantiser for our recursion.

Why this file exists. `experiments/generated_codebook_rho.py` showed that the
obvious way to make a lattice codebook resident -- take the 256 shortest E8
points and sign-flip them -- is a bad quantiser: rho^2 = 0.287, worse than the
scalar codebook's 0.1171, because the shortest E8 points are already close to
closed under sign flips, so the 2^16 labels collapse onto far fewer distinct
vectors. QuIP# solves exactly this, and the solution is the reason E8P is
called "padded".

The construction, from arXiv 2402.04396 Sec 4.2:

  * D8h = {x in Z^8 + 1/2 : 1^T x is even}, and (D8h - 1/4) u (D8h + 1/4)
    = E8 + 1/4, a shifted copy of E8 with the same packing density.
  * S = the 227 elements of |D8h| with norm <= sqrt(10), padded with 29
    elements of norm sqrt(12), giving |S| = 256. These are ABSOLUTE VALUES:
    one representative per sign orbit, which is what makes the sign flips
    generate distinct points instead of collapsing.
  * A 16-bit codeword is 8 bits for the entry of S, 7 bits of sign flips (the
    8th is recovered from parity), and 1 bit choosing +1/4 or -1/4.
    256 * 128 * 2 = 2^16.

The decoder therefore holds 256 int32 = 1 KiB, not 1 MiB.

This module is data-free throughout: S is a fixed function of the lattice, and
the single global scale is tuned against the KNOWN spherical marginal, never
against weights.

Encoding exploits the same structure, which is what makes a whole-model sweep
affordable: sign-matching is optimal for a fixed magnitude pattern, so the
search is over 256 magnitudes x 2 shifts rather than 2^16 codewords, with a
one-coordinate correction when the parity constraint is violated. `verify()`
checks that against brute force over the materialised 2^16 codebook.

    uv run python experiments/e8p_stage.py
"""

from __future__ import annotations

import argparse
import itertools
import math

import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def d8h_abs_source() -> torch.Tensor:
    """The 256-entry source codebook S of E8P: absolute values from |D8h|.

    Coordinates of D8h lie in Z + 1/2, so |D8h| entries lie in
    {0.5, 1.5, 2.5, 3.5}^8. We take all patterns with norm^2 <= 10 and pad with
    the shortest remaining ones (norm^2 = 12) up to 256, matching the paper's
    227 + 29 split.
    """
    vals = torch.tensor([0.5, 1.5, 2.5, 3.5])
    # 4^8 = 65536 magnitude patterns; enumerate and filter by norm.
    grids = torch.meshgrid(*([vals] * 8), indexing="ij")
    pts = torch.stack([g.reshape(-1) for g in grids], dim=1)
    nrm = pts.pow(2).sum(1)
    keep = pts[nrm <= 10.0 + 1e-6]
    pad = pts[(nrm > 10.0 + 1e-6) & (nrm <= 12.0 + 1e-6)]
    pad = pad[pad.pow(2).sum(1).argsort()]
    S = torch.cat([keep, pad], dim=0)[:256]
    assert S.shape == (256, 8), S.shape
    return S.to(DEV)


def parity_of(S: torch.Tensor) -> torch.Tensor:
    """Flip parity required for each magnitude pattern to land in D8h.

    The all-positive vector s has coordinate sum sum(s); D8h needs an even sum.
    Flipping coordinate i changes the sum by -2 s_i, and 2 s_i is an odd
    integer, so each flip changes the sum's parity. Hence the required number of
    flips has parity (sum(s) mod 2).
    """
    return (S.sum(1).round().long() % 2)


def build_full_codebook(S: torch.Tensor) -> torch.Tensor:
    """Materialise all 2^16 E8P codewords. Only used to verify the fast path."""
    par = parity_of(S)
    signs = torch.tensor(list(itertools.product([1.0, -1.0], repeat=8)),
                         device=S.device)                       # (256, 8)
    nflip = (signs < 0).sum(1)                                  # (256,)
    out = []
    for shift in (0.25, -0.25):
        for j in range(256):
            ok = (nflip % 2) == par[j]
            v = S[j].unsqueeze(0) * signs[ok] + shift           # (128, 8)
            out.append(v)
    cb = torch.cat(out, dim=0)
    assert cb.shape[0] == 1 << 16, cb.shape
    return cb.contiguous()


class E8PStage:
    """Nearest-E8P-codeword stage quantiser with one globally tuned scale."""

    def __init__(self, d: int, n_probe: int = 20000, seed: int = 0,
                 chunk: int = 16384):
        self.S = d8h_abs_source()                               # (256, 8)
        self.par = parity_of(self.S)                            # (256,)
        self.Ssq = self.S.pow(2).sum(1)                         # (256,)
        self.chunk = chunk
        self.name = "e8p-b2"
        g = torch.Generator(device=DEV).manual_seed(seed)
        probe = _sample_sphere(n_probe // (d // 8) + 1, d, g).reshape(-1, 8)
        lo, hi = 1e-3 / math.sqrt(d), 4.0 / math.sqrt(d)
        for _ in range(28):
            m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
            d1 = (probe - self._q(probe, m1)).pow(2).sum().item()
            d2 = (probe - self._q(probe, m2)).pow(2).sum().item()
            hi, lo = (m2, lo) if d1 < d2 else (hi, m1)
        self.scale = 0.5 * (lo + hi)

    def _nearest(self, x: torch.Tensor) -> torch.Tensor:
        """x: (m, 8) -> nearest E8P codeword, exactly, via the structure.

        For a fixed magnitude pattern s and shift c, the best signs are the
        signs of (x - c) coordinate-wise. If that violates the parity
        constraint, exactly one coordinate must be flipped, and the cheapest is
        the one minimising the increase 4|y_i| s_i.
        """
        best_d = None
        best_v = None
        for c in (0.25, -0.25):
            y = x - c                                            # (m, 8)
            a = y.abs()
            # ||a - s||^2 = ||a||^2 - 2 a.s + ||s||^2. ||a||^2 is constant
            # within a shift but NOT across the two shifts, so it must be kept:
            # dropping it makes the c = +1/4 vs c = -1/4 comparison meaningless.
            base = (a.pow(2).sum(1, keepdim=True)
                    + self.Ssq.unsqueeze(0) - 2.0 * (a @ self.S.T))  # (m, 256)
            # cheapest single flip for each (x, s)
            pen = 4.0 * (a.unsqueeze(2) * self.S.T.unsqueeze(0)).min(dim=1).values
            nneg = (y < 0).sum(1)                                # (m,)
            mism = ((nneg.unsqueeze(1) % 2) != self.par.unsqueeze(0))
            dist = base + torch.where(mism, pen, torch.zeros_like(pen))
            dmin, j = dist.min(dim=1)                            # (m,)
            s = self.S[j]                                        # (m, 8)
            sgn = torch.where(y < 0, -1.0, 1.0)
            need = mism.gather(1, j.unsqueeze(1)).squeeze(1)
            # flip the cheapest coordinate where parity demands it
            k = (a * s).argmin(dim=1)
            flip = torch.zeros_like(sgn, dtype=torch.bool)
            flip[torch.arange(x.shape[0], device=x.device), k] = True
            sgn = torch.where(flip & need.unsqueeze(1), -sgn, sgn)
            v = sgn * s + c
            if best_d is None:
                best_d, best_v = dmin, v
            else:
                take = dmin < best_d
                best_d = torch.where(take, dmin, best_d)
                best_v = torch.where(take.unsqueeze(1), v, best_v)
        return best_v

    def _q(self, blocks: torch.Tensor, scale: float) -> torch.Tensor:
        out = torch.empty_like(blocks)
        for i in range(0, blocks.shape[0], self.chunk):
            b = blocks[i : i + self.chunk] / scale
            out[i : i + self.chunk] = self._nearest(b) * scale
        return out

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        return self._q(u.reshape(-1, 8), self.scale).reshape(u.shape)


def _sample_sphere(n: int, d: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(n, d, generator=gen, device=DEV, dtype=torch.float32)
    return x / x.norm(dim=1, keepdim=True)


def verify(stage: E8PStage, n: int = 4096) -> tuple[float, float]:
    """Check the structured search against brute force over all 2^16 codewords."""
    cb = build_full_codebook(stage.S)
    cb_sq = cb.pow(2).sum(1)
    g = torch.Generator(device=DEV).manual_seed(7)
    x = _sample_sphere(n, 512, g).reshape(-1, 8) / stage.scale
    fast = stage._nearest(x)
    brute = torch.empty_like(x)
    step = 4096
    for i in range(0, x.shape[0], step):
        b = x[i : i + step]
        idx = (cb_sq.unsqueeze(0) - 2.0 * (b @ cb.T)).argmin(dim=1)
        brute[i : i + step] = cb[idx]
    df = (x - fast).pow(2).sum(1)
    db = (x - brute).pow(2).sum(1)
    return (df - db).abs().max().item(), (df.mean() / db.mean()).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--n", type=int, default=4096)
    args = ap.parse_args()

    st = E8PStage(args.d)
    print("# E8P as a data-free stage quantiser\n")
    print(f"# source codebook |S| = {st.S.shape[0]} magnitude patterns "
          f"({st.S.shape[0]*4/1024:.1f} KiB as int32), "
          f"2^16 codewords, tuned scale {st.scale:.5f}\n")

    dmax, dratio = verify(st)
    print(f"# structured search vs brute force over all 2^16 codewords: "
          f"max distance gap {dmax:.2e}, mean-distortion ratio {dratio:.6f}")
    print("# (1.000000 means the fast decoder is exactly optimal)\n")

    g = torch.Generator(device=DEV).manual_seed(1)
    u = _sample_sphere(args.n, args.d, g)
    r2 = (u - st(u)).pow(2).sum(1).mean().item()
    print("| stage codebook | table | rho^2 | rho |")
    print("|---|---:|---:|---:|")
    print(f"| E8P (QuIP#), 256 magnitudes + signs + shift | 1 KiB | {r2:.4f} | "
          f"{math.sqrt(r2):.4f} |")
    print("\n# For reference on the same footing (experiments/generated_codebook_rho.py):")
    print("#   E8 stored, all 2^16 points, 1 MiB : rho^2 = 0.0906")
    print("#   naive sign-flip generated, 4 KiB  : rho^2 = 0.2867")
    print("#   scalar Lloyd-Max (ours, shipped)  : rho^2 = 0.1171")


if __name__ == "__main__":
    main()
