"""The near-flat parameter c = max_j |v_j| sqrt(d) on real residual directions.

The paper's sharp bound on |rho_H^2(v) - rho^2| is stated for near-flat directions,
|v_j| <= c / sqrt(d), and reads

    2 * (3 c^2 / sqrt(d) + (13/10) c + 1 + 1/(4(d-1))) * spread(q)

(Threshold.abs_hadRate_sub_rhoSq_le_of_nearFlat).  The general route, stated in the
fourth-moment spikiness S(v) = sum_j v_j^4 instead, reads

    2 * spread(q) * (sqrt(d) * (15 S + 2 S^(1/4)) + sqrt(1 + 1/(4(d-1))))

(Threshold.abs_hadRate_sub_rhoSq_le_unconditional).  tab:spikiness measures d*S(v) and so
feeds the second.  It does not measure c, which is what the first consumes, and the first is
the sharper of the two by a wide margin at flat directions.  This script measures c on the
same layers, passes and protocol as experiments/hadamard_spikiness.py, and prints both bounds
at the measured values so that the proof can be read against the weights it is meant to cover.

A bound over a layer must hold for every group in it, so the max column is the one that
decides; the p99 column is reported because it says how heavy the tail is, not because any
theorem consumes it.

    uv run python experiments/nearflat_parameter.py path/to/clay-v1.5.ckpt
"""

from __future__ import annotations

import argparse

import torch

from apexquant.codebook.apex_beta_levels import apex_beta_levels

from hadamard_spikiness import DEV, GROUP_SIZE, LAYERS, PASSES, BITS, residual_directions

# The proof first beats the free bound at a Haar-typical direction at d = 1831, so the sweep
# brackets that: 512 is what we ship, 2048 is the first power of two above the threshold.
SWEEP_GROUPS = (512, 1024, 2048, 4096)
# Pass 0 is the raw weight direction, which every arm agrees is spiky; the question the sweep
# asks is about the regime the decay law governs, which starts at pass 1.
FIRST_GOVERNED_PASS = 1

# The free bound |rho_H^2 - rho^2| <= 1 holds whenever zero is a codeword, so a printed
# bound at or above this says nothing that was not already free.
FREE_BOUND = 1.0
# Coefficients of the two proved bounds, as the Lean statements carry them.
NEARFLAT_QUAD = 3.0
NEARFLAT_LIN = 13.0 / 10.0
GENERAL_S = 15.0
GENERAL_S_QUARTER = 2.0


def nearflat_bound(c: float, d: int, spread: float) -> float:
    """Threshold.abs_hadRate_sub_rhoSq_le_of_nearFlat, evaluated."""
    inner = NEARFLAT_QUAD * c * c / d**0.5 + NEARFLAT_LIN * c + 1.0 + 1.0 / (4.0 * (d - 1))
    return 2.0 * inner * spread


def general_bound(s: float, d: int, spread: float) -> float:
    """Threshold.abs_hadRate_sub_rhoSq_le_unconditional, evaluated."""
    leg1 = d**0.5 * (GENERAL_S * s + GENERAL_S_QUARTER * s**0.25)
    leg2 = (1.0 + 1.0 / (4.0 * (d - 1))) ** 0.5
    return 2.0 * spread * (leg1 + leg2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--bits", type=int, default=BITS)
    args = ap.parse_args()
    torch.manual_seed(0)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    levels = apex_beta_levels(args.bits, GROUP_SIZE, device=DEV)
    d = GROUP_SIZE
    spread = float(levels.max() - levels.min())

    print("# The near-flat parameter c = max_j |v_j| sqrt(d), and the bounds it feeds")
    print(f"# Clay v1.5, {args.bits}-bit apex codebook, group size d={d}, {PASSES} passes")
    print(f"# spread(q) = {spread:.5f} = {spread * d**0.5:.4f}/sqrt(d); "
          f"free bound = {FREE_BOUND}")
    print("# c = 1 is the exactly-flat direction; a Haar direction has c near "
          f"sqrt(2 ln d) = {(2 * torch.log(torch.tensor(float(d)))) ** 0.5:.2f}\n")
    print("| layer | pass | mean c | p99 c | max c | bound at max c | bound at mean d*S |")
    print("|---|---|---|---|---|---|---|")
    for name, key in LAYERS.items():
        for p, v in enumerate(residual_directions(sd[key], levels, seed=0)):
            c = v.abs().max(dim=-1).values * d**0.5
            s_mean = float(v.pow(4).sum(-1).mean())
            nf = nearflat_bound(float(c.max()), d, spread)
            gen = general_bound(s_mean, d, spread)
            print(f"| {name} | {p} | {float(c.mean()):.2f} | "
                  f"{float(c.quantile(0.99)):.2f} | {float(c.max()):.2f} | "
                  f"{nf:.2f} | {gen:.2f} |")

    print("\n## The same measurement as the group size grows, passes "
          f"{FIRST_GOVERNED_PASS} onward\n")
    print("# Only layers whose fan-in the group divides are pooled at each d. Zero padding a\n"
          "# narrow layer up to a wide group inflates c by exactly sqrt(group/fan-in), so a\n"
          "# padded layer would measure the padding and not the weights.")
    print("| d | layers | spread(q) | sqrt(2 ln d) | mean c | max c | "
          "bound at mean c | bound at max c |")
    print("|---|---|---|---|---|---|---|---|")
    for g in SWEEP_GROUPS:
        lv = apex_beta_levels(args.bits, g, device=DEV)
        spr = float(lv.max() - lv.min())
        keys = [k for k in LAYERS.values() if sd[k].shape[1] % g == 0]
        if not keys:
            raise ValueError(f"no layer has a fan-in divisible by {g}")
        cs = []
        for key in keys:
            dirs = residual_directions(sd[key], lv, seed=0, group=g)
            for v in dirs[FIRST_GOVERNED_PASS:]:
                cs.append(v.abs().max(dim=-1).values * g**0.5)
        c = torch.cat(cs)
        c_mean, c_max = float(c.mean()), float(c.max())
        typical = (2.0 * torch.log(torch.tensor(float(g)))) ** 0.5
        print(f"| {g} | {len(keys)} | {spr:.5f} | {float(typical):.2f} | {c_mean:.2f} | "
              f"{c_max:.2f} | {nearflat_bound(c_mean, g, spr):.3f} | "
              f"{nearflat_bound(c_max, g, spr):.3f} |")


if __name__ == "__main__":
    main()
