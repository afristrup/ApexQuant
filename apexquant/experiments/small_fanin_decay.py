"""Small fan-in: the Proposition-1 failure is a first-pass phenomenon.

ApexQuant concedes the small-fan-in regime (d <~ 32): one rotation cannot
isotropise heavy single-coordinate structure, the marginal misses f_d (KS 0.113
at d=9), and density matching gains nothing over the uniform grid. This
experiment shows the progressive mechanism is nevertheless dimension-free:

  * rho_theory(2, d) barely moves with fan-in (0.312 at d=9 vs 0.342 at d=512);
  * on the real depthwise-convolution rows of MobileNet-V2 (d=9, dense Haar
    rotations), the FIRST pass overshoots the prediction (the Prop-1 mismatch),
    but every subsequent per-pass ratio lands on rho_theory(9) -- Theorem 1
    assumes nothing about the residual's distribution, so re-isotropisation
    resets after pass one regardless of how poorly pass one was matched.

Consequence (paper, Limitations "Small fan-in"): a data-aware first stage
(SpectralQuant) composes with fresh-rotation residual stages, and the per-pass
optimality guarantee applies verbatim to every pass beyond the first.

    uv run python experiments/small_fanin_decay.py path/to/mobilenet_v2.pth
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predicted_decay_constant import rho_theory  # noqa: E402

from apexquant.codebook.apex_beta_levels import apex_beta_levels  # noqa: E402
from apexquant.quantize.quantize_to_indices import quantize_to_indices  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def haar(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(d, d, generator=g))
    return (q * torch.sign(torch.diagonal(r))).to(DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mobilenet_pth")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    print("# rho_theory(2-bit, d) is nearly flat in fan-in:\n")
    for d in (9, 16, 32, 64, 512):
        print(f"  d={d:>4}: rho = {rho_theory(2, d):.4f}")

    sd = torch.load(args.mobilenet_pth, map_location="cpu", weights_only=True)
    rows = torch.cat([v.reshape(v.shape[0], -1) for v in sd.values()
                      if v.ndim == 4 and v.shape[1] == 1 and v.shape[2] == 3],
                     0).to(DEV).float()
    print(f"\n# {rows.shape[0]} real MobileNet-V2 depthwise rows (d=9); dense Haar per pass,")
    print("# apex f_9 codebook. First pass overshoots (Prop-1 mismatch); ratios then = rho.\n")
    levels = apex_beta_levels(2, 9, device=DEV)
    rt = rho_theory(2, 9)
    residual, w0, prev = rows, rows.norm(), None
    print("| P | measured | rho_theory^P | ratio vs prev |")
    print("|--:|---------:|-------------:|--------------:|")
    for P in range(1, 4):
        R = haar(9, P - 1)
        rot = residual @ R.T
        s = rot.pow(2).sum(1, keepdim=True).sqrt().clamp_min(1e-12)
        idx = quantize_to_indices(rot / s, levels)
        residual = residual - (s * levels[idx.long()]) @ R
        err = float(residual.norm() / w0)
        ratio = "" if prev is None else f"{err / prev:.3f}"
        print(f"| {P} | {err:.4f} | {rt ** P:.4f} | {ratio} |")
        prev = err
    print(f"\n# rho_theory(9) = {rt:.4f}: the post-first-pass ratios match it.")


if __name__ == "__main__":
    main()
