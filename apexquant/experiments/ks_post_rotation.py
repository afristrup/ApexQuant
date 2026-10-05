"""Kolmogorov-Smirnov distance of the post-rotation coordinates against f_d.

`appendix_RHT.tex` proves a KS bound, eq:ksbound, and measures the fourth-moment
gap that drives it, but never measured the distance itself on real weights, so
the appendix carried a `\\needs{}` rather than a number. This closes it.

What is measured. For each pass the deployed path rotates the current residual
by a fresh randomized Hadamard transform and divides each group by its own L2
norm, so every group becomes a point on the unit sphere S^{gs-1}. The coordinate
marginal of a uniform point on that sphere is f_d, with density proportional to
(1 - u^2)^{(d-3)/2}, and it is the density the codebook is matched to. We pool
the normalized coordinates over every row and group of a layer and compare their
empirical CDF against the exact f_d CDF, because that pooled distribution is
what the quantizer actually sees. This is an estimate of the distance for the one
rotation that is deployed, not of the expectation over rotations that
eq:ksbound bounds; the bound is an upper envelope on the latter.

The unrotated arm is the control: raw weight directions are spiky, so their
normalized coordinates should be far from f_d, and one rotation should close it.

    uv run python experiments/ks_post_rotation.py path/to/clay-v1.5.ckpt
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from scipy.special import betainc
from scipy.stats import kstest

from ablations import DEV, _rotate, load_layers

GROUP = 512
PASSES = 3
# Enough coordinates that the KS statistic's own sampling noise, which falls as
# 1/sqrt(N), sits well below the distances we report. At N = 2^20 that noise is
# about 1.2e-3.
SAMPLES = 1 << 20
# The constants of eq:ksbound, for a spread unit direction.
BOUND_SQRT, BOUND_LIN = 8.2, 10.2


def f_d_cdf(u, d: int):
    """Exact CDF of one coordinate of a uniform point on the unit sphere S^{d-1}.

    (u + 1) / 2 is Beta((d-1)/2, (d-1)/2), so the regularized incomplete beta
    gives the CDF in closed form with no quadrature.
    """
    half = (d - 1) / 2.0
    return betainc(half, half, (u + 1.0) / 2.0)


def ks_against_f_d(coords: torch.Tensor, d: int) -> float:
    """KS distance between the pooled coordinates and f_d."""
    flat = coords.reshape(-1)
    if flat.numel() > SAMPLES:
        pick = torch.randperm(flat.numel(), device=flat.device)[:SAMPLES]
        flat = flat[pick]
    return float(kstest(flat.cpu().numpy(), lambda u: f_d_cdf(u, d)).statistic)


def sphere_coords(w: torch.Tensor, passes: int, rotate: bool):
    """Normalized coordinates entering the codebook at each pass.

    Yields one tensor per pass. With rotate=False the rotation is skipped and
    the raw residual direction is normalized instead, which is the control.
    """
    out_f, in_f = w.shape
    x = F.pad(w, (0, (-in_f) % GROUP)).view(out_f, -1, GROUP)
    levels = _apex_levels(GROUP)
    residual = x
    for p in range(passes):
        rot = _rotate(residual, p, "hadamard", True) if rotate else residual
        scale = rot.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
        unit = rot / scale
        yield unit
        recon = scale * levels[_quantize(unit, levels).long()]
        residual = residual - (_rotate(recon, p, "hadamard", False) if rotate else recon)


def _apex_levels(gs: int):
    from apexquant.codebook.apex_beta_levels import apex_beta_levels
    return apex_beta_levels(2, gs, device=DEV)


def _quantize(x, levels):
    from apexquant.quantize.quantize_to_indices import quantize_to_indices
    return quantize_to_indices(x, levels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    layers = load_layers(args.checkpoint)

    bound = BOUND_SQRT / GROUP**0.5 + BOUND_LIN / GROUP
    print(f"# KS distance of the post-rotation coordinates against f_d, d={GROUP}")
    print(f"# {len(layers)} real Clay v1.5 layers, {SAMPLES} pooled coordinates per cell")
    print(f"# eq:ksbound evaluates to {bound:.3f} at this d, which is why it is vacuous\n")
    print("| condition | " + " | ".join(f"pass {p + 1}" for p in range(PASSES)) + " |")
    print("|---|" + "---|" * PASSES)

    for label, rotate in (("rotated (deployed)", True), ("unrotated (control)", False)):
        per_pass = [[] for _ in range(PASSES)]
        for _, w in layers:
            for p, unit in enumerate(sphere_coords(w, PASSES, rotate)):
                per_pass[p].append(ks_against_f_d(unit, GROUP))
        cells = []
        for vals in per_pass:
            t = torch.tensor(vals)
            cells.append(f"{t.mean():.4f} +/- {t.std():.4f}")
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
