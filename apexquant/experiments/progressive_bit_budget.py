"""Progressive 2-bit quantization matches single-pass at the same bit budget.

If the density-matched codebook is optimal at every residual pass (see
``reisotropization_theorem.py``), then P passes of a b-bit codebook should spend
b*P index bits and buy the same reconstruction quality as ONE pass of a b*P-bit
codebook -- i.e. a progressive quantizer that reaches high-bit quality while only
ever storing 2-bit indices (the regime where fused low-bit kernels and INT storage
apply). This measures reconstruction rel-error at matched effective bits on REAL
Clay v1.5 weights.

Effective bits = bits*passes + (scale_bits / group_size)*passes; at group_size=512
the fp16 scale adds ~0.03 bit/pass, so effective bits ~= bits*passes.

    uv run python experiments/progressive_bit_budget.py path/to/clay-v1.5.ckpt
"""

from __future__ import annotations

import argparse

import torch

from apexquant.codebook.apex_beta_levels import apex_beta_levels
from apexquant.quantize.quantize_to_indices import quantize_to_indices
from apexquant.rotation.inverse_random_rotation import inverse_random_rotation
from apexquant.rotation.random_rotation import random_rotation

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rel_error(weight, bits, gs, passes, seed=0):
    w = weight.to(DEV, torch.float32)
    out_f, in_f = w.shape
    pad = (-in_f) % gs
    w = torch.nn.functional.pad(w, (0, pad)).view(out_f, -1, gs)
    levels = apex_beta_levels(bits, gs, device=DEV)
    residual, w0 = w, w.norm()
    for p in range(passes):
        rot = random_rotation(residual, seed + p)
        scale = rot.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
        idx = quantize_to_indices(rot / scale, levels)
        residual = residual - inverse_random_rotation(scale * levels[idx.long()], seed + p)
    return float(residual.norm() / w0)


def eff_bits(bits, gs, passes, scale_bits=16):
    return (bits + scale_bits / gs) * passes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--gs", type=int, default=512)
    args = ap.parse_args()
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    pick = {
        "qkv (3072x1024)": "model.encoder.transformer.layers.0.0.to_qkv.weight",
        "fc1 (4096x1024)": "model.encoder.transformer.layers.11.1.net.1.weight",
        "fc2 (1024x4096)": "model.encoder.transformer.layers.23.1.net.3.weight",
    }
    # (label, bits, passes) grouped by effective-bit budget.
    configs = [
        ("2-bit x1", 2, 1), ("2-bit x2", 2, 2), ("4-bit x1", 4, 1),
        ("2-bit x3", 2, 3), ("2-bit x4", 2, 4), ("4-bit x2", 4, 2), ("8-bit x1", 8, 1),
    ]
    print("# Progressive 2-bit vs single-pass at matched bit budget (rel-error on real Clay)\n")
    print(f"# group_size={args.gs}  device={DEV}\n")
    header = "| config    | eff.bits | " + " | ".join(pick) + " |"
    print(header)
    print("|" + "---|" * (len(pick) + 2))
    for label, bits, passes in configs:
        eb = eff_bits(bits, args.gs, passes)
        errs = [rel_error(sd[key], bits, args.gs, passes) for key in pick.values()]
        print(f"| {label:<9} | {eb:>7.2f}  | " + " | ".join(f"{e:.4f}" for e in errs) + " |")

    print("\n# Read: compare rows at equal eff.bits. 2-bit x2 (~4.06b) vs 4-bit x1 (~4.03b);")
    print("#       2-bit x4 (~8.13b) vs 4-bit x2 (~8.06b) vs 8-bit x1 (~8.03b).")

    # The theorem's fingerprint: since every pass re-isotropizes to the same f_d,
    # each pass should shrink the residual by the SAME factor rho = single-pass
    # rel-error, giving a geometric err(P) = rho^P. Show the per-pass ratio.
    print("\n# Geometric-decay check (2-bit): err(p+1)/err(p) should equal the single-pass err.\n")
    print(f"| layer | {' | '.join(f'err(p{p})' for p in range(1, 5))} | ratios p2/p1,p3/p2,p4/p3 |")
    print("|" + "---|" * 6)
    for key_label, key in pick.items():
        e = [rel_error(sd[key], 2, args.gs, p) for p in range(1, 5)]
        ratios = [e[i + 1] / e[i] for i in range(3)]
        print(f"| {key_label} | " + " | ".join(f"{v:.4f}" for v in e)
              + " | " + ", ".join(f"{r:.3f}" for r in ratios) + " |")


if __name__ == "__main__":
    main()
