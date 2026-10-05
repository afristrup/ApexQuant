"""The geometric-decay constant rho is a data-free property of the codebook.

progressive_bit_budget.py showed err(P) = rho^P on real Clay, with rho = the
single-pass rel-error. Here we PREDICT rho in closed form from the ApexQuant
codebook and the spherical marginal f_d, with no data at all.

Derivation. After a random rotation and row L2-normalization, a weight row is a
unit vector on S^{d-1}; the rotation is orthogonal so it preserves norms. Hence

    rel_err^2 = ||W - W_hat||^2 / ||W||^2 = sum_i (u_i - q(u_i))^2 / sum_i u_i^2,

and taking expectations over the sphere (each coord ~ f_d),

    rho^2 = E_{f_d}[(u - q(u))^2] / E_{f_d}[u^2]
          = codebook distortion / variance  =  normalized distortion of q under f_d.

Both integrals are 1-D quadratures over the known density -> rho is predicted
data-free. We compare rho_theory to the single-pass rel-error and the measured
geometric ratio on real Clay weights.

    uv run python experiments/predicted_decay_constant.py path/to/clay-v1.5.ckpt
"""

from __future__ import annotations

import argparse

import torch

from apexquant.codebook.apex_beta_levels import apex_beta_levels
from apexquant.quantize.quantize_to_indices import quantize_to_indices
from apexquant.rotation.inverse_random_rotation import inverse_random_rotation
from apexquant.rotation.random_rotation import random_rotation

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rho_theory(bits: int, d: int, grid: int = 200001) -> float:
    """Predicted decay constant: sqrt(distortion / variance) of the apex codebook
    under the spherical marginal f_d -- purely from the density, no data."""
    u = torch.linspace(-1.0, 1.0, grid, dtype=torch.float64)
    w = (1.0 - u * u).clamp_min(0.0) ** ((d - 3) / 2.0)
    w = w / w.sum()
    levels = apex_beta_levels(bits, d).double()
    # Nearest level for each grid point (cell boundaries are level midpoints).
    bnd = 0.5 * (levels[1:] + levels[:-1])
    q = levels[torch.bucketize(u, bnd)]
    distortion = float((w * (u - q) ** 2).sum())
    variance = float((w * u * u).sum())          # equals 1/d in the exact limit
    return (distortion / variance) ** 0.5


def measured_single_pass(weight, bits, gs, seed=0) -> float:
    w = weight.to(DEV, torch.float32)
    out_f, in_f = w.shape
    pad = (-in_f) % gs
    w = torch.nn.functional.pad(w, (0, pad)).view(out_f, -1, gs)
    levels = apex_beta_levels(bits, gs, device=DEV)
    rot = random_rotation(w, seed)
    scale = rot.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
    idx = quantize_to_indices(rot / scale, levels)
    residual = w - inverse_random_rotation(scale * levels[idx.long()], seed)
    return float(residual.norm() / w.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    args = ap.parse_args()
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    keys = {
        "qkv (d=1024)": "model.encoder.transformer.layers.0.0.to_qkv.weight",
        "fc1 (d=1024)": "model.encoder.transformer.layers.11.1.net.1.weight",
        "fc2 (d=4096)": "model.encoder.transformer.layers.23.1.net.3.weight",
    }
    print("# Data-free prediction of the geometric-decay constant rho\n")
    print("# rho_theory = sqrt(distortion/variance) of the apex codebook under f_d (no data).\n")
    print("| bits | gs=d | rho_theory | measured single-pass (qkv/fc1/fc2) |")
    print("|-----:|-----:|-----------:|-------------------------------------|")
    for bits in (2, 3, 4):
        for gs in (512,):
            rt = rho_theory(bits, gs)
            meas = [measured_single_pass(sd[k], bits, gs) for k in keys.values()]
            print(f"| {bits} | {gs} | {rt:.4f} | "
                  + " / ".join(f"{m:.4f}" for m in meas) + " |")

    print("\n# rho_theory should match the measured single-pass error AND the per-pass")
    print("# geometric ratio (progressive_bit_budget.py). At 2-bit both were ~0.342.")

    # Show the predicted vs realised multi-pass curve for 2-bit, gs=512, qkv.
    rt = rho_theory(2, 512)
    print(f"\n# Predicted 2-bit curve err(P) = rho_theory^P with rho_theory={rt:.4f}:")
    print("| P | predicted rho^P | measured (qkv) |")
    print("|--:|----------------:|---------------:|")
    w = sd[keys["qkv (d=1024)"]].to(DEV, torch.float32)
    out_f, in_f = w.shape
    ww = torch.nn.functional.pad(w, (0, (-in_f) % 512)).view(out_f, -1, 512)
    levels = apex_beta_levels(2, 512, device=DEV)
    residual, w0 = ww, ww.norm()
    for P in range(1, 5):
        rot = random_rotation(residual, P - 1)
        scale = rot.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
        idx = quantize_to_indices(rot / scale, levels)
        residual = residual - inverse_random_rotation(scale * levels[idx.long()], P - 1)
        print(f"| {P} | {rt ** P:.4f} | {float(residual.norm() / w0):.4f} |")


if __name__ == "__main__":
    main()
