"""Does the *generated* lattice codebook keep the distortion of the stored one?

experiments/lattice_decode_cost.py shows that the decode cost of a lattice stage
is set by whether the codebook is stored (1 MiB, an L2 gather per coordinate) or
generated (4 KiB, L1-resident), not by the lattice: the generated arm runs at
0.8-1.3x the scalar decode against the stored arm's 1.3-3.9x.

That is only worth reporting if the generated codebook is still a good quantiser.
A cheap decode of a bad codebook is not an argument. So this script measures
rho^2 for the generated construction on the same footing as tab:stages: the same
uniform directions, the same single globally tuned scale, the same rate of two
bits per coordinate.

The construction under test, matching the kernel: a 16-bit code splits into
8 bits selecting one of 256 source vectors and 8 sign bits, one per coordinate,
so the effective codebook is {+-s_i} over the 256 shortest E8 points -- 2^16
vectors in 8 dimensions, generated rather than stored.

    uv run python experiments/generated_codebook_rho.py
"""

from __future__ import annotations

import argparse
import math

import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_sphere(n: int, d: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(n, d, generator=gen, device=DEV, dtype=torch.float32)
    return x / x.norm(dim=1, keepdim=True)


def e8_points(n_codewords: int) -> torch.Tensor:
    """The `n_codewords` shortest E8 points (same enumeration as tab:stages)."""
    rng = torch.arange(-4, 4, device=DEV, dtype=torch.int8)
    grids = torch.meshgrid(*([rng] * 8), indexing="ij")
    pts = torch.stack([g.reshape(-1) for g in grids], dim=1).to(torch.float32)
    even = pts[(pts.sum(1).to(torch.int64) % 2) == 0]
    e8 = torch.cat([even, even + 0.5], dim=0)
    nrm = e8.pow(2).sum(1)
    keep = nrm <= 12.0 + 1e-6
    e8, nrm = e8[keep], nrm[keep]
    return e8[nrm.argsort()[:n_codewords]].contiguous()


def generated_codebook() -> torch.Tensor:
    """{+-s_i} over the 256 shortest E8 points: 2^16 vectors, 4 KiB of state."""
    src = e8_points(256)                                   # (256, 8)
    signs = torch.arange(256, device=DEV)
    bits = (signs.unsqueeze(1) >> torch.arange(8, device=DEV)) & 1
    mult = 1.0 - 2.0 * bits.float()                        # (256, 8)
    return (src.unsqueeze(1) * mult.unsqueeze(0)).reshape(-1, 8).contiguous()


def rho_sq(cb: torch.Tensor, d: int, n: int, gen: torch.Generator) -> float:
    """E||u - Q(u)||^2 for u uniform on S^{d-1}, single globally tuned scale."""
    cb_sq = cb.pow(2).sum(1)
    n_cw = cb.shape[0]

    def quantise(blocks: torch.Tensor, scale: float) -> torch.Tensor:
        out = torch.empty_like(blocks)
        step = max(1, (1 << 24) // max(1, n_cw))
        for i in range(0, blocks.shape[0], step):
            b = blocks[i : i + step] / scale
            idx = (cb_sq.unsqueeze(0) - 2.0 * (b @ cb.T)).argmin(dim=1)
            out[i : i + step] = cb[idx] * scale
        return out

    probe = sample_sphere(20000 // (d // 8) + 1, d, gen).reshape(-1, 8)
    lo, hi = 1e-3 / math.sqrt(d), 4.0 / math.sqrt(d)
    for _ in range(28):
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        d1 = (probe - quantise(probe, m1)).pow(2).sum().item()
        d2 = (probe - quantise(probe, m2)).pow(2).sum().item()
        hi, lo = (m2, lo) if d1 < d2 else (hi, m1)
    scale = 0.5 * (lo + hi)

    u = sample_sphere(n, d, gen)
    err = (u.reshape(-1, 8) - quantise(u.reshape(-1, 8), scale)).pow(2).sum(1)
    return err.reshape(n, -1).sum(1).mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--n", type=int, default=4096)
    args = ap.parse_args()

    gen = torch.Generator(device=DEV).manual_seed(0)
    stored = e8_points(1 << 16)
    generated = generated_codebook()

    print("# Distortion of the generated vs stored lattice codebook\n")
    print(f"# d={args.d}, 2 bits/coordinate, {args.n} uniform directions, "
          "one globally tuned scale (data-free).\n")
    print("| stage codebook | state held | rho^2 | rho |")
    print("|---|---:|---:|---:|")
    for name, cb, state in [
        ("$E_8$ stored, $2^{16}$ codewords", stored, "1 MiB"),
        ("$E_8$ generated, 256 sources + signs", generated, "4 KiB"),
    ]:
        r2 = rho_sq(cb, args.d, args.n, gen)
        print(f"| {name} | {state} | {r2:.4f} | {math.sqrt(r2):.4f} |",
              flush=True)
    print("\n# Scalar Lloyd-Max at the same rate is 0.1171 (tab:stages).")


if __name__ == "__main__":
    main()
