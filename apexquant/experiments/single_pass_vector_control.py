"""The fair control: a single wide pass with a VECTOR stage, at the same rate.

Why this exists. On Mistral-7B at ~4 effective bits, progressive refinement with
an E8P stage (2 bits/coord, two passes, 5.418) beats a single wide 4-bit pass
(5.441), reversing the concession the paper makes. But that comparison changes
two things at once: it pits a *vector* stage against a *scalar* one AND two
narrow passes against one wide pass. The win could be entirely the stage.

This script builds the missing cell of the 2x2 design. At 4 bits per coordinate
a vector stage needs 2^(4d) codewords, so d=8 is out (2^32), but d=4 is exactly
2^16 -- the same codebook size as our E8 stage, over four dimensions instead of
eight. The natural lattice there is D4, the checkerboard lattice, which is the
optimal packing in four dimensions as E8 is in eight.

    stage                            rate        err after the pass
    scalar Lloyd-Max, 1 pass         4 bits      rho
    scalar Lloyd-Max, 2 passes       2+2 bits    rho^2
    D4 lattice, 1 pass               4 bits      rho          <- this file
    E8P lattice, 2 passes            2+2 bits    rho^2

Everything stays data-free: D4 is a fixed lattice, and the single global scale is
tuned against the known spherical marginal, never against weights.

Because err(P) = rho^P is what the theory predicts and what tab:stages verifies
to 0.3%, comparing rho directly settles the question at the theory level without
a model run: if the single wide vector pass has smaller err than the two-pass
vector artifact, the reversal was the stage, not the refinement.

    uv run python experiments/single_pass_vector_control.py
"""

from __future__ import annotations

import argparse
import math

import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_sphere(n: int, d: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(n, d, generator=gen, device=DEV, dtype=torch.float32)
    return x / x.norm(dim=1, keepdim=True)


def d4_codebook(n_codewords: int = 1 << 16) -> torch.Tensor:
    """The `n_codewords` shortest points of D4 = {x in Z^4 : sum(x) even}.

    D4 is the densest lattice packing in four dimensions, the four-dimensional
    analogue of E8. A box of +-13 comfortably contains more than 2^16 points of
    even coordinate sum.
    """
    r = torch.arange(-13, 14, device=DEV, dtype=torch.float32)
    g = torch.meshgrid(*([r] * 4), indexing="ij")
    pts = torch.stack([x.reshape(-1) for x in g], dim=1)
    even = pts[(pts.sum(1).to(torch.int64) % 2) == 0]
    nrm = even.pow(2).sum(1)
    assert even.shape[0] >= n_codewords, even.shape
    return even[nrm.argsort()[:n_codewords]].contiguous()


class LatticeStage:
    """Nearest-codeword stage over a finite lattice, one globally tuned scale."""

    def __init__(self, cb: torch.Tensor, dim: int, d: int, name: str,
                 chunk: int = 2048, n_probe: int = 20000, seed: int = 0):
        self.cb = cb
        self.cb_sq = cb.pow(2).sum(1)
        self.dim = dim
        self.chunk = chunk
        self.name = name
        gen = torch.Generator(device=DEV).manual_seed(seed)
        probe = sample_sphere(n_probe // (d // dim) + 1, d, gen).reshape(-1, dim)
        lo, hi = 1e-4 / math.sqrt(d), 4.0 / math.sqrt(d)
        for _ in range(30):
            m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
            d1 = (probe - self._q(probe, m1)).pow(2).sum().item()
            d2 = (probe - self._q(probe, m2)).pow(2).sum().item()
            hi, lo = (m2, lo) if d1 < d2 else (hi, m1)
        self.scale = 0.5 * (lo + hi)

    def _q(self, blocks: torch.Tensor, scale: float) -> torch.Tensor:
        out = torch.empty_like(blocks)
        for i in range(0, blocks.shape[0], self.chunk):
            b = blocks[i : i + self.chunk] / scale
            idx = (self.cb_sq.unsqueeze(0) - 2.0 * (b @ self.cb.T)).argmin(dim=1)
            out[i : i + self.chunk] = self.cb[idx] * scale
        return out

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        return self._q(u.reshape(-1, self.dim), self.scale).reshape(u.shape)


def rho_sq(Q, d: int, n: int, seed: int = 1) -> float:
    gen = torch.Generator(device=DEV).manual_seed(seed)
    u = sample_sphere(n, d, gen)
    return (u - Q(u)).pow(2).sum(1).mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--n", type=int, default=2048)
    args = ap.parse_args()

    print("# The fair single-pass vector control at ~4 effective bits\n")
    print(f"# d={args.d}, {args.n} uniform directions, one globally tuned scale "
          "(fitted to the known spherical marginal, not to weights).\n")

    d4 = LatticeStage(d4_codebook(), 4, args.d, "D4, 2^16 codewords over 4 dims")
    r2_d4 = rho_sq(d4, args.d, args.n)
    err_d4 = math.sqrt(r2_d4)

    # published / already-measured numbers on the identical footing
    r2_scalar4 = 0.00946      # scalar Lloyd-Max at 4 bits (bit-width sweep)
    r2_e8p = 0.0912           # E8P at 2 bits (experiments/e8p_stage.py)
    r2_scalar2 = 0.1171       # scalar Lloyd-Max at 2 bits (tab:stages)

    rows = [
        ("scalar Lloyd--Max, 1 pass", 4, 1, r2_scalar4),
        ("D4 lattice, 1 pass",        4, 1, r2_d4),
        ("scalar Lloyd--Max, 2 passes", 2, 2, r2_scalar2),
        ("E8P lattice, 2 passes",     2, 2, r2_e8p),
    ]
    print("| stage | bits/coord | passes | eff. bits | rho^2 | rho | err = rho^P |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, b, P, r2 in rows:
        rho = math.sqrt(r2)
        print(f"| {name} | {b} | {P} | {P*(b+16/512):.3f} | {r2:.5f} | "
              f"{rho:.4f} | {rho**P:.4f} |")

    err_e8p2 = math.sqrt(r2_e8p) ** 2
    err_sc4 = math.sqrt(r2_scalar4)
    print()
    print(f"# D4 single pass at 4 bits:        err = {err_d4:.4f}")
    print(f"# E8P two passes at 2 bits:        err = {err_e8p2:.4f}")
    print(f"# scalar single pass at 4 bits:    err = {err_sc4:.4f}")
    print()
    if err_d4 < err_e8p2:
        print("# VERDICT: the fair vector-vs-vector control RESTORES the single")
        print("# wide pass. The Mistral reversal is attributable to the stage,")
        print("# not to progressive refinement.")
    else:
        print("# VERDICT: progressive refinement wins even against a matched")
        print("# vector single pass. The reversal survives the control.")


if __name__ == "__main__":
    main()
