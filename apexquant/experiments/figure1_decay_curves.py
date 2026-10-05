"""Every number Figure 1 panel B plots, measured on one fixed set of Clay layers.

The panel needs three things the repo did not have in one place:

  1. the measured decay curve at b = 2, 3 and 4 over passes 1..4, so all three
     bit-widths have full curves rather than one having a single lonely marker;
  2. the closed-form rho_theory for each b, which is what the dashed lines are
     supposed to be. The figure currently plots 0.1900 and 0.0985 as the "closed-form
     prediction" for b = 3 and b = 4, and reuses the same two literals as the
     "measured" markers. Neither is the closed form, which is 0.1855 and 0.0972;
  3. the fixed-rotation counterfactual, the direct test of re-isotropization.

It also measures the Haar-versus-Hadamard gap per pass, which Section 3 asserts is
small but which nothing in the repo had measured beyond a single pass count.

Everything runs through `ablations.reconstruct`, so these numbers are produced by the
same code path as the recorded A1 study and are comparable to it.

    uv run python experiments/figure1_decay_curves.py path/to/clay-v1.5.ckpt

"""

from __future__ import annotations

import argparse

import torch

from ablations import load_layers, rel_err
from predicted_decay_constant import rho_theory

BITS = (2, 3, 4)
PASSES = (1, 2, 3, 4)
GROUP = 512
BASE = dict(codebook="apex", gs=GROUP, rotation="hadamard", fresh=True)


def curve(layers, **kw):
    """Mean and std of the relative error across layers, per pass count."""
    out = []
    for P in PASSES:
        vals = torch.tensor([rel_err(w, passes=P, **kw) for _, w in layers])
        out.append((vals.mean().item(), vals.std().item()))
    return out


def show(label, pts):
    body = "  ".join(f"{m:.4f}+/-{s:.4f}" for m, s in pts)
    ratios = "  ".join(f"{pts[i][0] / pts[i - 1][0]:.3f}" for i in range(1, len(pts)))
    print(f"| {label:28s} | {body} | {ratios} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    layers = load_layers(args.checkpoint)
    names = ", ".join(n for n, _ in layers)
    print(f"# Figure 1 panel B, on {len(layers)} real Clay v1.5 layers, group size {GROUP}")
    print(f"# layers: {names}\n")

    print("## Closed-form prediction (no data read)\n")
    print("| b | rho_theory | rho_theory^2 | rho^P for P=1..4 |")
    print("|--:|-----------:|-------------:|------------------|")
    rho = {}
    for b in BITS:
        r = rho_theory(b, GROUP)
        rho[b] = r
        pows = "  ".join(f"{r ** P:.4f}" for P in PASSES)
        print(f"| {b} | {r:.4f} | {r * r:.4f} | {pows} |")

    print("\n## Measured, fresh rotation per pass (mean +/- std over layers)\n")
    print("| condition                    | P=1, P=2, P=3, P=4                              "
          "        | per-pass ratios |")
    print("|------------------------------|--------------------------------------------------"
          "----|-----------------|")
    for b in BITS:
        show(f"b={b}, fresh Hadamard", curve(layers, bits=b, **BASE))

    print("\n## Counterfactual: rotation held fixed across passes (b=2)\n")
    print("| condition                    | P=1, P=2, P=3, P=4                              "
          "        | per-pass ratios |")
    print("|------------------------------|--------------------------------------------------"
          "----|-----------------|")
    show("b=2, fixed rotation", curve(layers, bits=2, **{**BASE, "fresh": False}))

    print("\n## Haar versus Hadamard, per pass (b=2, fresh)\n")
    had = curve(layers, bits=2, **BASE)
    dense = curve(layers, bits=2, **{**BASE, "rotation": "dense"})
    print("| P | Hadamard | dense random-orthogonal | relative gap |")
    print("|--:|---------:|------------------------:|-------------:|")
    for P, (h, _), (d, _) in zip(PASSES, had, dense):
        print(f"| {P} | {h:.4f} | {d:.4f} | {abs(h - d) / d:.2e} |")


if __name__ == "__main__":
    main()
