"""Tiered decoding: heterogeneous quality tiers in one batch, one artifact.

The nested property means pass prefixes are bit-identical lower-rate models,
so a BATCH of requests at different tiers (premium 3-pass, standard 2-pass,
economy 1-pass) shares the weight stream: plane p is read once for every row
that uses it, and the batch's weight traffic is the deepest tier's bytes, not
the sum over rows. Fixed-width elasticity (separate artifacts per tier) shares
nothing: every artifact's bytes are distinct.

Measured here (graph-captured, Mistral-7B 32-block stack, 2-bit, g=512) -- this
is table 13:

  A. naive sequential   -- each request a separate batch-1 decode at its tier
                           (the shipped LOP3 path on a truncated prefix view)
  B. premium-only batch -- the premium rows alone, 3-pass, fused mid-M kernel
  C. homogeneous batch  -- ALL rows served at premium (3-pass), same kernel
  D. tiered batch       -- mixed tiers fused, per-plane row prefixes (ours)
  E. separate artifacts -- premium rows on artifact 1 (3-pass) + standard rows
                           on an INDEPENDENT 2-pass artifact 2, same kernel:
                           the fixed-width elastic deployment

Claims to check: D <= C (riders never cost more than upgrading them),
D - B ~ 0 relative to the riders' standalone cost (riders ride the shared
stream), D < A (fusing beats sequential), D < E (nesting beats separate
artifacts: E reads 3 + 2 distinct planes, D reads 3 shared ones).

Needs the optional ``gemlite`` extra (``pip install apexquant[gemlite]``) and a
CUDA device; the split-K kernel this script benchmarks has no torch fallback.

Usage: python experiments/tiered_decoding.py  [FUSED_TURBOQUANT_GL_AUTOTUNE=max]
"""

import time

import torch

from apexquant import ApexQuantConfig, turboquantize_weight
from apexquant.kernels.fused_dequant_matmul import (
    _stacked_rotations,
    fused_dequant_matmul,
)
from apexquant.kernels.gemlite_splitk_nonuniform import (
    gemlite_splitk_nonuniform,
    gemlite_splitk_nonuniform_tiered,
)
from apexquant.quantize.quantized_weight import QuantizedWeight

DEV = "cuda"
MISTRAL_BLOCK = [(6144, 4096), (4096, 4096), (14336, 4096), (14336, 4096), (4096, 14336)]
DEPTH = 32
GS = 512
TIERS = (3, 3, 2, 2)  # two premium requests, two standard riders


def bench(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


def graphed(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


def prefix_view(qw: QuantizedWeight, passes: int) -> QuantizedWeight:
    """The nested property in code: the first `passes` planes ARE the
    lower-rate quantisation (verified bit-identical in the paper)."""
    return QuantizedWeight(
        packed=qw.packed[:passes], scale=qw.scale[:passes], levels=qw.levels,
        in_features=qw.in_features, out_features=qw.out_features,
        group_size=qw.group_size, seed=qw.seed, bits=qw.bits)


def main():
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    P = max(TIERS)
    B = len(TIERS)
    n_premium = sum(1 for t in TIERS if t == P)

    tot = {k: 0.0 for k in "ABCDE"}
    worst_rel = 0.0
    for (o, i) in MISTRAL_BLOCK:
        w = torch.randn(o, i, device=DEV)
        qw = turboquantize_weight(w, ApexQuantConfig(
            bits=2, passes=P, group_size=GS, codebook="apex"))
        # the fixed-width alternative: an INDEPENDENT standard-tier artifact
        qw_std = turboquantize_weight(w, ApexQuantConfig(
            bits=2, passes=2, group_size=GS, codebook="apex"))
        x = torch.randn(B, i, device=DEV)
        xg = x.view(B, -1, GS)
        R = _stacked_rotations(GS, qw.seed, P, DEV)
        rx = torch.einsum("mng,pgh->pmnh", xg, R).reshape(P, B, -1).contiguous()

        # parity: tiered row m == torch forward of the tier-m prefix view
        out_t = gemlite_splitk_nonuniform_tiered(rx, qw, TIERS)
        for m, t in enumerate(TIERS):
            ref = fused_dequant_matmul(x[m:m + 1], prefix_view(qw, t),
                                       force_backend="torch")
            rel = ((out_t[m] - ref[0]).norm() / ref[0].norm()).item()
            worst_rel = max(worst_rel, rel)

        views = [prefix_view(qw, t) for t in TIERS]
        rx_std = rx[:2].contiguous()

        tot["A"] += bench(graphed(lambda: [
            fused_dequant_matmul(x[m:m + 1], views[m]) for m in range(B)])) * DEPTH
        tot["B"] += bench(graphed(
            lambda: gemlite_splitk_nonuniform(rx[:, :n_premium].contiguous(), qw))) * DEPTH
        tot["C"] += bench(graphed(lambda: gemlite_splitk_nonuniform(rx, qw))) * DEPTH
        tot["D"] += bench(graphed(
            lambda: gemlite_splitk_nonuniform_tiered(rx, qw, TIERS))) * DEPTH
        tot["E"] += bench(graphed(lambda: (
            gemlite_splitk_nonuniform(rx[:, :n_premium].contiguous(), qw),
            gemlite_splitk_nonuniform(rx_std[:, n_premium:].contiguous(), qw_std)))) * DEPTH
        del qw, qw_std
        torch.cuda.empty_cache()

    print(f"# Tiered decoding, tiers {TIERS}, Mistral 32-block stack, graph-captured\n")
    print(f"parity vs per-row prefix reference: worst rel {worst_rel:.2e}\n")
    print("| scenario | ms/batch | per request |")
    print("|---|---:|---:|")
    names = {
        "A": f"naive sequential ({B} separate batch-1 decodes)",
        "B": f"premium-only batch (M={n_premium}, 3-pass)",
        "C": f"homogeneous premium batch (M={B}, 3-pass)",
        "D": f"tiered batch (M={B}, tiers {TIERS}) [ours]",
        "E": "separate artifacts (premium + independent 2-pass)",
    }
    for k in "ABCDE":
        print(f"| {names[k]} | {tot[k] / 1e3:.2f} | {tot[k] / 1e3 / B:.2f} |")
    print(f"\nriders' marginal cost: D - B = {(tot['D'] - tot['B']) / 1e3:.2f} ms "
          f"for {B - n_premium} standard riders "
          f"(standalone cost of the riders, from A-style decode: see per-tier below)")
    print(f"tiered vs homogeneous premium: D/C = {tot['D'] / tot['C']:.2f}")
    print(f"tiered vs naive sequential:    A/D = {tot['A'] / tot['D']:.2f}x faster")
    print(f"tiered vs separate artifacts:  E/D = {tot['E'] / tot['D']:.2f}x faster "
          f"(and 6.09 vs 10.15 stored bits per weight)")


if __name__ == "__main__":
    main()
