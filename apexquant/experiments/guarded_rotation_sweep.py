"""Per-pass error ratio of one Hadamard, guarded Hadamard and dense Haar frames.

Inputs are unit directions of growing spikiness. For each, the scalar two-bit
stage runs PASSES fresh-frame passes and the script prints the mean per-pass
ratio |r_{p+1}|^2 / |r_p|^2, the normalized third moment of the input, the
share of (group, pass) pairs where the guard fired, and the bound of the
paper's scalar Hadamard proposition.

    python experiments/guarded_rotation_sweep.py
"""

from __future__ import annotations

import torch

from apexquant.codebook.apex_beta_levels import apex_beta_levels
from apexquant.quantize.quantize_to_indices import quantize_to_indices
from apexquant.rotation.guarded_rotation import guarded_rotation
from apexquant.rotation.inverse_guarded_rotation import inverse_guarded_rotation
from apexquant.rotation.normalized_third_moment import normalized_third_moment

D = 512
BITS = 2
PASSES = 4
SAMPLES = 2048
THRESHOLD = 4.0
SEED = 0


def directions(kind: str, gen: torch.Generator) -> torch.Tensor:
    if kind == "gaussian":
        x = torch.randn(SAMPLES, D, generator=gen)
    elif kind.startswith("sparse"):
        k = int(kind.split("-")[1])
        x = torch.zeros(SAMPLES, D)
        cols = torch.rand(SAMPLES, D, generator=gen).argsort(dim=-1)[:, :k]
        x.scatter_(1, cols, 1.0)
    else:
        noise = float(kind.split("-")[1])
        x = noise * torch.randn(SAMPLES, D, generator=gen) / D ** 0.5
        x[:, 0] += 1.0
    return x / x.norm(dim=-1, keepdim=True)


def haar_frames(gen: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(64, D, D, generator=gen))
    return q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)


def run(x: torch.Tensor, levels: torch.Tensor, frame: str, gen: torch.Generator) -> tuple[float, float]:
    residual, ratios, fired = x.clone(), [], []
    for p in range(PASSES):
        before = residual.pow(2).sum(-1)
        if frame == "haar":
            q = haar_frames(gen).repeat(SAMPLES // 64, 1, 1)
            rotated = torch.einsum("nij,nj->ni", q, residual)
        else:
            threshold = THRESHOLD if frame == "guarded" else float("inf")
            rotated, flags = guarded_rotation(residual, SEED + p, threshold)
            fired.append(flags.float().mean())
        scale = rotated.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        recon = scale * levels[quantize_to_indices(rotated / scale, levels).long()]
        if frame == "haar":
            residual = residual - torch.einsum("nij,ni->nj", q, recon)
        else:
            residual = residual - inverse_guarded_rotation(recon, SEED + p, flags)
        ratios.append((residual.pow(2).sum(-1) / before).mean())
    return torch.stack(ratios).mean().item(), (torch.stack(fired).mean().item() if fired else 0.0)


def main() -> None:
    gen = torch.Generator().manual_seed(SEED)
    levels = apex_beta_levels(BITS, D, device="cpu")
    spread = ((levels[-1] - levels[0]) * D ** 0.5).item()
    print(f"{'input':<12}{'m3*sqrt(d)':>11}{'bound':>8}{'one RHT':>9}{'guarded':>9}{'fired':>7}{'Haar':>8}")
    for kind in ("gaussian", "sparse-64", "sparse-8", "sparse-2", "axis-1.0", "axis-0.3", "axis-0.0"):
        x = directions(kind, gen)
        m3 = normalized_third_moment(x).mean().item()
        bound = spread * (3 * m3 / D ** 0.5 + (2 / D) ** 0.5)
        one, _ = run(x, levels, "one", gen)
        guarded, fired = run(x, levels, "guarded", gen)
        haar, _ = run(x, levels, "haar", gen)
        print(f"{kind:<12}{m3:>11.2f}{bound:>8.2f}{one:>9.4f}{guarded:>9.4f}{fired:>7.2f}{haar:>8.4f}")


if __name__ == "__main__":
    main()
