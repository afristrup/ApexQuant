"""Spikiness of real residual directions and the certified Hadamard fourth-moment gap.

The Lean development (thms/frags/36_hadamard.lean, 37_fourthmoment.lean) proves, for a unit
direction v and the deployed rotation R = H diag(eps) with fresh uniform signs,

    E (R v)_i^2 = 1/d,      E (R v)_i^4 = (3 - 2 S(v)) / d^2,      S(v) = sum_j v_j^4,

while the spherical marginal f_d has fourth moment 3/(d(d+2)); the fourth-moment gap between a
Hadamard coordinate and f_d is exactly (6/(d+2) - 2 S(v))/d^2, between -2/d^2 and 4/d^3.  The
gap is governed by the spikiness S(v) of the residual direction: S = 1/d for a flat direction
(gap ~ 4/d^3), S ~ 3/d for a Gaussian-like one (gap ~ -4/d^3), S = 1 for a coordinate vector
(gap ~ -2/d^2).  This script measures S(v) on real Clay v1.5 residual directions at every pass,
checks the certified identity against sampled sign patterns, and measures the per-pass rate
rho_H^2(v) = E_eps |R v - q(R v)|^2 against the same rate under a dense Haar rotation, the
quantity behind the appendix's "three parts in a thousand".

    uv run python experiments/hadamard_spikiness.py path/to/clay-v1.5.ckpt
"""

from __future__ import annotations

import argparse

import torch

from apexquant.codebook.apex_beta_levels import apex_beta_levels
from apexquant.quantize.quantize_to_indices import quantize_to_indices
from apexquant.rotation.inverse_random_rotation import inverse_random_rotation
from apexquant.rotation.random_rotation import random_rotation
from apexquant.rotation.walsh_hadamard_transform import walsh_hadamard_transform

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIGN_SAMPLES = 128
SPHERE_SAMPLES = 200_000
CHECK_GROUPS = 512
CHECK_SIGN_SAMPLES = 512
GROUP_SIZE = 512
PASSES = 3
BITS = 2
LAYERS = {
    "layers.0 qkv": "model.encoder.transformer.layers.0.0.to_qkv.weight",
    "layers.11 fc1 (GELU-facing)": "model.encoder.transformer.layers.11.1.net.1.weight",
    "layers.11 fc2": "model.encoder.transformer.layers.11.1.net.3.weight",
    "layers.23 fc1 (GELU-facing)": "model.encoder.transformer.layers.23.1.net.1.weight",
    "layers.23 fc2": "model.encoder.transformer.layers.23.1.net.3.weight",
}


def independent_signs(shape: torch.Size, gen: torch.Generator) -> torch.Tensor:
    """An independent uniform sign pattern for every group."""
    return torch.randint(0, 2, shape, generator=gen, device=DEV).float() * 2 - 1


def pass_rate(v: torch.Tensor, levels: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Squared quantisation error |x - q(x)|^2 of the rotated unit directions x (per group)."""
    idx = quantize_to_indices(x, levels)
    return (x - levels[idx.long()]).pow(2).sum(-1)


def residual_directions(weight: torch.Tensor, levels: torch.Tensor, seed: int,
                        group: int = GROUP_SIZE):
    """Unit residual directions before each pass of the deployed pipeline.

    ``group`` defaults to the published GROUP_SIZE, so every number this module
    printed before it became a parameter is unchanged.
    """
    w = weight.to(DEV, torch.float32)
    out_f, in_f = w.shape
    pad = (-in_f) % group
    w = torch.nn.functional.pad(w, (0, pad)).view(out_f, -1, group).reshape(-1, group)
    residual = w
    dirs = []
    for p in range(PASSES):
        norms = residual.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        dirs.append(residual / norms)
        rot = random_rotation(residual, seed + p)
        scale = rot.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        idx = quantize_to_indices(rot / scale, levels)
        recon = inverse_random_rotation(scale * levels[idx.long()], seed + p)
        residual = residual - recon
    return dirs


def spikiness_stats(v: torch.Tensor) -> dict[str, float]:
    d = v.shape[-1]
    s = v.pow(4).sum(-1) * d
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
    }


def check_identity(v: torch.Tensor, gen: torch.Generator) -> float:
    """Max relative deviation of the sampled fourth moment from the certified formula."""
    d = v.shape[-1]
    sub = v[torch.randperm(v.shape[0], generator=gen, device=DEV)[:CHECK_GROUPS]]
    predicted = (3.0 - 2.0 * sub.pow(4).sum(-1)) / d**2
    acc = torch.zeros(sub.shape[0], device=DEV)
    for k in range(CHECK_SIGN_SAMPLES):
        x = random_rotation(sub, 10_000 + k)
        acc += x.pow(4).mean(-1)
    sampled = acc / CHECK_SIGN_SAMPLES
    return float(((sampled - predicted) / predicted).abs().max())


def hadamard_rate(v: torch.Tensor, levels: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Per-group rate rho_H^2(v), averaged over independent sign patterns per group."""
    had = torch.zeros(v.shape[0], device=DEV)
    for _ in range(SIGN_SAMPLES):
        x = walsh_hadamard_transform(v * independent_signs(v.shape, gen))
        had += pass_rate(v, levels, x)
    return had / SIGN_SAMPLES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    # The published table is b=2. The fourth-moment gap is a property of the
    # direction and so is bit-width independent, but the per-pass RATE is not:
    # it is an expectation of the stage's error, and a finer codebook resolves
    # the difference between the Hadamard coordinate law and f_d more sharply.
    ap.add_argument("--bits", type=int, default=BITS)
    args = ap.parse_args()
    torch.manual_seed(0)
    gen = torch.Generator(device=DEV).manual_seed(0)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    levels = apex_beta_levels(args.bits, GROUP_SIZE, device=DEV)
    d = GROUP_SIZE
    sphere4 = 3.0 / (d * (d + 2))

    # The Haar rate is the same for every unit direction, by invariance: the quadrature rho^2.
    ref = torch.randn(SPHERE_SAMPLES, d, device=DEV)
    ref = ref / ref.norm(dim=-1, keepdim=True)
    sphere_rates = pass_rate(ref, levels, ref)
    rho_sq = float(sphere_rates.mean())
    rho_sq_err = float(sphere_rates.std() / SPHERE_SAMPLES**0.5)

    print("# Spikiness of residual directions and the certified Hadamard gap")
    print(f"# Clay v1.5, {args.bits}-bit apex codebook, group size d={d}, {PASSES} passes")
    print(f"# Haar rate rho^2 = {rho_sq:.5f} +- {rho_sq_err:.5f} (Monte Carlo, "
          f"{SPHERE_SAMPLES} sphere draws; the same for every direction, by invariance)")
    print(f"# sphere fourth moment 3/(d(d+2)) = {sphere4:.3e}; certified gap is")
    print("#   (6/(d+2) - 2 S)/d^2 in [-2/d^2, 4/d^3] = "
          f"[{-2 / d**2:.2e}, {4 / d**3:.2e}]; d*S = 1 flat, ~3 Gaussian-like, d spike\n")

    print("## (A) Spikiness d*S(v) = d * sum_j v_j^4 of the unit residual directions\n")
    print("| layer | pass | mean | median | p99 | max | mean gap / sphere moment |")
    print("|---|---|---|---|---|---|---|")
    all_dirs = {}
    for label, key in LAYERS.items():
        dirs = residual_directions(sd[key], levels, seed=0)
        all_dirs[label] = dirs
        for p, v in enumerate(dirs):
            st = spikiness_stats(v)
            gap = (6.0 / (d + 2) - 2.0 * v.pow(4).sum(-1)) / d**2
            rel = float((gap / sphere4).mean())
            print(f"| {label} | {p} | {st['mean']:.3f} | {st['median']:.3f} | "
                  f"{st['p99']:.3f} | {st['max']:.1f} | {rel:+.2e} |")

    print("\n## (B) Certified fourth-moment identity vs sampled sign patterns")
    print(f"#     max relative deviation over {CHECK_GROUPS} groups, "
          f"{CHECK_SIGN_SAMPLES} sign patterns each (pass 0 directions)\n")
    for label, dirs in all_dirs.items():
        print(f"| {label} | {check_identity(dirs[0], gen):.3e} |")

    print("\n## (C) Per-pass rate under the deployed rotation vs the Haar rate rho^2")
    print(f"#     rho_H^2(v) averaged over {SIGN_SAMPLES} independent sign patterns per group; "
          "the Haar rate is rho^2 for every direction\n")
    print("| layer | pass | mean rho_H^2 | rho_H^2/rho^2 - 1 | top 5% spiky | bottom 50% | "
          "flattest 5% |")
    print("|---|---|---|---|---|---|---|")
    for label, dirs in all_dirs.items():
        for p, v in enumerate(dirs):
            had = hadamard_rate(v, levels, gen)
            s = v.pow(4).sum(-1)
            top = s >= s.quantile(0.95)
            bot = s <= s.quantile(0.50)
            flat = s <= s.quantile(0.05)
            r_all = float(had.mean()) / rho_sq - 1.0
            r_top = float(had[top].mean()) / rho_sq - 1.0
            r_bot = float(had[bot].mean()) / rho_sq - 1.0
            r_flat = float(had[flat].mean()) / rho_sq - 1.0
            print(f"| {label} | {p} | {float(had.mean()):.5f} | {r_all:+.4f} | {r_top:+.4f} | "
                  f"{r_bot:+.4f} | {r_flat:+.4f} |")


if __name__ == "__main__":
    main()
