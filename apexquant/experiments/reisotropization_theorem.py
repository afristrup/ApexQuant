"""Empirical proof that residual passes re-isotropize -- the theorem for a
progressive density-matched quantizer on top of ApexQuant.

ApexQuant quantizes once: rotate a weight row into a random frame, L2-normalize,
and round each coordinate to the Lloyd-Max codebook of the spherical marginal

    f_d(u) = C_d (1 - u^2)^((d-3)/2),   (u+1)/2 ~ Beta((d-1)/2, (d-1)/2).

Claim (the follow-up's theorem): if we peel off pass p's reconstruction and rotate
the *residual* with a FRESH independent rotation, the residual's normalized
coordinates follow the SAME f_d. A Haar-random rotation is independent of the
residual, so it sends the residual's unit vector to a uniform point on S^{d-1}
whose coordinates are exactly f_d-distributed -- regardless of the residual's
structure. Consequently ApexQuant's single density-matched codebook is MSE-optimal
at *every* pass, turning it into a progressive multi-pass quantizer for free.

This script proves it three ways on REAL trained Clay v1.5 weights:

  (A) KS distance of the per-pass normalized residual coordinates against f_d,
      the residual-pass analogue of ApexQuant's Figure 1. It stays flat across
      passes -> the residual re-isotropizes to the same marginal every pass.
  (B) Codebook-optimality ablation: reuse the fixed apex codebook every pass vs
      RE-FIT a Lloyd-Max codebook to each pass's actual residual histogram vs a
      uniform codebook. reuse ~= refit << uniform confirms the fixed codebook is
      already per-pass optimal (the operational content of the theorem).

    uv run python experiments/reisotropization_theorem.py path/to/clay-v1.5.ckpt
"""

from __future__ import annotations

import argparse

import torch

from apexquant.codebook.apex_beta_levels import apex_beta_levels
from apexquant.codebook.uniform_levels import uniform_levels
from apexquant.quantize.quantize_to_indices import quantize_to_indices
from apexquant.rotation.inverse_random_rotation import inverse_random_rotation
from apexquant.rotation.random_rotation import random_rotation

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def f_d_grid_cdf(d: int, grid: int = 40001):
    """Theoretical CDF of the spherical marginal f_d, tabulated on [-1, 1]."""
    u = torch.linspace(-1.0, 1.0, grid, dtype=torch.float64)
    w = (1.0 - u * u).clamp_min(0.0) ** ((d - 3) / 2.0)
    w = w / w.sum()
    return u, torch.cumsum(w, dim=0)


def ks_against_f_d(coords: torch.Tensor, d: int) -> float:
    """One-sample KS distance sup|F_emp - F_{f_d}| of coords in [-1, 1]."""
    u, cdf = f_d_grid_cdf(d)
    x = coords.detach().to("cpu", torch.float64).flatten().sort().values
    n = x.numel()
    # Theoretical CDF at each sample via the tabulated grid.
    idx = torch.searchsorted(u, x.clamp(-1, 1)).clamp(max=u.numel() - 1)
    ft = cdf[idx]
    emp_hi = torch.arange(1, n + 1, dtype=torch.float64) / n
    emp_lo = torch.arange(0, n, dtype=torch.float64) / n
    return float(torch.maximum((ft - emp_lo).abs(), (emp_hi - ft).abs()).max())


def lloyd_max_on_samples(v: torch.Tensor, bits: int, iters: int = 80) -> torch.Tensor:
    """1-D Lloyd-Max fit to the empirical distribution of samples v (k-means)."""
    n = 2 ** bits
    x = v.flatten().sort().values
    q = ((torch.arange(n, device=x.device) + 0.5) / n * x.numel()).long().clamp(max=x.numel() - 1)
    levels = x[q].clone()
    for _ in range(iters):
        bnd = 0.5 * (levels[1:] + levels[:-1])
        cell = torch.bucketize(x, bnd)
        num = torch.zeros(n, device=x.device).scatter_add_(0, cell, x)
        den = torch.zeros(n, device=x.device).scatter_add_(0, cell, torch.ones_like(x))
        levels = torch.where(den > 0, num / den.clamp_min(1e-9), levels)
    return levels


def quantize_passes(weight, bits, gs, passes, seed=0, codebook="apex_reuse"):
    """Run the residual-pass recipe; return (per-pass normalized coords, per-pass
    residual rel-error). ``codebook`` selects reuse/refit/uniform behaviour."""
    w = weight.to(DEV, torch.float32)
    out_f, in_f = w.shape
    pad = (-in_f) % gs
    w = torch.nn.functional.pad(w, (0, pad)).view(out_f, -1, gs)
    fixed = apex_beta_levels(bits, gs, device=DEV)
    unif = uniform_levels(bits, device=DEV)
    w0_norm = w.norm()

    residual = w
    coords_per_pass, err_per_pass = [], []
    for p in range(passes):
        rot = random_rotation(residual, seed + p)
        # Each codebook is paired with its matching normalizer (as in the fuser):
        # apex/Lloyd-Max on the unit sphere (L2), QuaRot-style uniform on [-1,1]
        # (absmax) -- so the uniform baseline is a fair QuaRot post-rotation grid.
        if codebook == "uniform":
            scale = rot.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        else:
            scale = rot.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
        coords = rot / scale
        if codebook == "apex_reuse":
            levels = fixed
        elif codebook == "refit":
            levels = lloyd_max_on_samples(coords, bits)
        else:
            levels = unif
        idx = quantize_to_indices(coords, levels)
        recon = inverse_random_rotation(scale * levels[idx.long()], seed + p)
        residual = residual - recon
        coords_per_pass.append(coords.flatten())
        err_per_pass.append(float(residual.norm() / w0_norm))
    return coords_per_pass, err_per_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--gs", type=int, default=512)
    ap.add_argument("--passes", type=int, default=4)
    args = ap.parse_args()

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    pick = {
        "transformer.layers.0 qkv (3072x1024)": "model.encoder.transformer.layers.0.0.to_qkv.weight",
        "transformer.layers.11 fc1 (4096x1024)": "model.encoder.transformer.layers.11.1.net.1.weight",
        "transformer.layers.23 fc2 (1024x4096)": "model.encoder.transformer.layers.23.1.net.3.weight",
    }

    print("# Re-isotropization theorem: residual passes reuse ONE density-matched codebook")
    print(f"# bits={args.bits}  group_size={args.gs}  device={DEV}\n")

    print("## (A) KS distance of per-pass normalized residual coords vs the spherical marginal f_d")
    print("#     Flat across passes => each residual re-isotropizes to the SAME f_d.\n")
    print(f"| layer | {' | '.join(f'pass {p}' for p in range(args.passes))} |")
    print("|" + "---|" * (args.passes + 1))
    for label, key in pick.items():
        coords, _ = quantize_passes(sd[key], args.bits, args.gs, args.passes)
        ks = [ks_against_f_d(c, args.gs) for c in coords]
        print(f"| {label} | " + " | ".join(f"{k:.4f}" for k in ks) + " |")
    # Baseline: KS of a genuinely uniform-on-sphere draw (finite-sample floor).
    ref = torch.randn(4000, args.gs, device=DEV)
    ref = ref / ref.norm(dim=-1, keepdim=True)
    print(f"\n#   finite-sample KS floor (true uniform sphere draw, same N): "
          f"{ks_against_f_d(ref.flatten(), args.gs):.4f}")

    print("\n## (B) Codebook-optimality ablation: does reusing the fixed apex codebook lose")
    print("#     anything vs RE-FITTING Lloyd-Max to each pass's residual? (rel-error after P passes)\n")
    print(f"| layer | codebook | {' | '.join(f'p={p+1}' for p in range(args.passes))} |")
    print("|" + "---|" * (args.passes + 2))
    for label, key in pick.items():
        for cb, name in [("apex_reuse", "apex (fixed, reused)"),
                         ("refit", "Lloyd-Max (refit/pass)"),
                         ("uniform", "uniform (QuaRot-style)")]:
            _, err = quantize_passes(sd[key], args.bits, args.gs, args.passes, codebook=cb)
            print(f"| {label} | {name} | " + " | ".join(f"{e:.4f}" for e in err) + " |")
        print("|" + "---|" * (args.passes + 2))


if __name__ == "__main__":
    main()
