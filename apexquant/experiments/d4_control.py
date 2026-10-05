"""The missing cell of the 2x2: one wide pass with a vector stage.

progressive_vector_stage_llm.py shows that on Mistral-7B, progressive refinement
with an E8P stage beats the single wide 4-bit pass at a matched budget (5.418
against 5.441). But that changes two things at once: it swaps the stage AND it
splits the rate across passes. The win may be entirely the stage.

The control fixes the index width at 16 bits and asks how best to spend it:

    shape 8 coordinates twice   E8P at 2 bits/coord, 2 passes -> err = rho^2
    shape 4 coordinates once    D4  at 4 bits/coord, 1 pass   -> err = rho

D4 = {x in Z^4 : sum(x) even} is the densest lattice packing in four dimensions,
as E8 is in eight, so this is the fair vector-stage single-pass comparator at
2^16 codewords. Both are data-free: fixed lattices, one globally tuned scale
each, fitted to the known spherical marginal and never to weights.

    uv run python experiments/d4_control.py
"""

from __future__ import annotations

import argparse
import math

import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_sphere(n: int, d: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(n, d, generator=gen, device=DEV, dtype=torch.float32)
    return x / x.norm(dim=1, keepdim=True)


def d4_codebook(n_codewords: int) -> torch.Tensor:
    """The `n_codewords` shortest points of D4 = {x in Z^4 : sum(x) even}."""
    rng = torch.arange(-12, 13, device=DEV, dtype=torch.float32)
    grids = torch.meshgrid(*([rng] * 4), indexing="ij")
    pts = torch.stack([g.reshape(-1) for g in grids], dim=1)
    keep = pts[(pts.sum(1).to(torch.int64) % 2) == 0]
    nrm = keep.pow(2).sum(1)
    assert keep.shape[0] >= n_codewords, f"only {keep.shape[0]} D4 points"
    return keep[nrm.argsort()[:n_codewords]].contiguous()


def rho_sq(cb: torch.Tensor, d: int, blk: int, n: int, gen: torch.Generator,
           chunk: int = 4096) -> float:
    """E||u - Q(u)||^2 for u uniform on S^{d-1}, one globally tuned scale."""
    cb_sq = cb.pow(2).sum(1)

    def q(blocks: torch.Tensor, scale: float) -> torch.Tensor:
        out = torch.empty_like(blocks)
        for i in range(0, blocks.shape[0], chunk):
            b = blocks[i : i + chunk] / scale
            idx = (cb_sq.unsqueeze(0) - 2.0 * (b @ cb.T)).argmin(dim=1)
            out[i : i + chunk] = cb[idx] * scale
        return out

    probe = sample_sphere(20000 // (d // blk) + 1, d, gen).reshape(-1, blk)
    lo, hi = 1e-4 / math.sqrt(d), 4.0 / math.sqrt(d)
    for _ in range(30):
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        d1 = (probe - q(probe, m1)).pow(2).sum().item()
        d2 = (probe - q(probe, m2)).pow(2).sum().item()
        hi, lo = (m2, lo) if d1 < d2 else (hi, m1)
    scale = 0.5 * (lo + hi)

    u = sample_sphere(n, d, gen)
    e = (u.reshape(-1, blk) - q(u.reshape(-1, blk), scale)).pow(2).sum(1)
    return e.reshape(n, -1).sum(1).mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--n", type=int, default=4096)
    args = ap.parse_args()

    gen = torch.Generator(device=DEV).manual_seed(3)
    cb = d4_codebook(1 << 16)
    r2 = rho_sq(cb, args.d, 4, args.n, gen)
    rho = math.sqrt(r2)

    e8p_r2 = 0.0912          # e8p_stage.py, verified exactly optimal
    scalar_b4_r2 = 0.00946   # bit-width sweep, tab:stagebits

    print("# The missing cell: one wide pass with a vector stage\n")
    print(f"# D4, 2^16 codewords over 4 dims = 4 bits/coordinate, d={args.d}, "
          f"{args.n} uniform directions, one tuned scale (data-free).\n")
    print(f"D4  rho^2 = {r2:.5f}   rho = {rho:.5f}\n")
    print("| stage | bits/coord | passes | eff. bits | err = rho^P |")
    print("|---|---:|---:|---:|---:|")
    print(f"| E8P (8 dims), 2 passes   | 2 | 2 | 4.062 | {e8p_r2:.4f} |")
    print(f"| D4 (4 dims), 1 pass      | 4 | 1 | 4.031 | {rho:.4f} |")
    print(f"| scalar Lloyd-Max, 1 pass | 4 | 1 | 4.031 | {math.sqrt(scalar_b4_r2):.4f} |")
    print(f"| scalar Lloyd-Max, 2 pass | 2 | 2 | 4.062 | {0.1171:.4f} |")
    print()
    if e8p_r2 < rho:
        print(f"# VERDICT: splitting wins. E8P x2 ({e8p_r2:.4f}) beats D4 x1 ({rho:.4f}).")
    else:
        print(f"# VERDICT: splitting does NOT win. D4 x1 ({rho:.4f}) beats or ties "
              f"E8P x2 ({e8p_r2:.4f}). The control claim FAILS and must be retracted.")


if __name__ == "__main__":
    main()
