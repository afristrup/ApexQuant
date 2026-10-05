"""Tiered decoding: the mix/batch-size sweep and the strongest fixed-width baseline.

Answers the two open questions on the tiered-batch result (table 13):

1. WINDOW. "Riders are nearly free" was measured at one operating point
   (M=4, tiers 3,3,2,2). How do the rider ratio and the tiered-vs-all-premium
   gap move as the mix shifts (all-premium -> all-standard at fixed M) and as
   the batch grows toward the compute-bound regime (M = 4..128, 50/50 mix)?

2. STRONGEST BASELINE. The separate-artifacts comparison is the naive
   fixed-width strategy. The strongest one is "serve everyone at premium and
   do not bother tiering": all rows at 3 passes, one artifact, same kernel.
   Tiering can only win here by the arithmetic it skips, so the honest
   question is where along the M axis that saving becomes material.

All timings graph-captured on the Mistral-7B 32-block stack, direct kernel
calls (the dispatcher's M<=64 policy cap does not apply to the sweep).
Autotune scope is whatever FUSED_TURBOQUANT_GL_AUTOTUNE says (default fast).

Needs the optional ``gemlite`` extra (``pip install apexquant[gemlite]``) and a
CUDA device; the split-K kernel this script benchmarks has no torch fallback.
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
P = 3


def bench(fn, iters=100, warmup=30):
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
    return QuantizedWeight(
        packed=qw.packed[:passes], scale=qw.scale[:passes], levels=qw.levels,
        in_features=qw.in_features, out_features=qw.out_features,
        group_size=qw.group_size, seed=qw.seed, bits=qw.bits)


def make_layers():
    layers = []
    for (o, i) in MISTRAL_BLOCK:
        w = torch.randn(o, i, device=DEV)
        qw = turboquantize_weight(w, ApexQuantConfig(
            bits=2, passes=P, group_size=GS, codebook="apex"))
        layers.append((o, i, qw))
    return layers


def stack_time(layers, fn_of_layer):
    total = 0.0
    for (o, i, qw) in layers:
        total += bench(graphed(fn_of_layer(o, i, qw))) * DEPTH
    return total / 1e3  # ms


def rx_for(qw, i, M):
    x = torch.randn(M, i, device=DEV)
    R = _stacked_rotations(GS, qw.seed, P, DEV)
    return torch.einsum("mng,pgh->pmnh", x.view(M, -1, GS), R).reshape(P, M, -1).contiguous()


def main():
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    layers = make_layers()

    # batch-1 standard-tier token: the standalone rider cost (shipped decode path)
    def solo_std(o, i, qw):
        pv = prefix_view(qw, 2)
        x = torch.randn(1, i, device=DEV)
        return lambda: fused_dequant_matmul(x, pv)
    t_solo_std = stack_time(layers, solo_std)
    print(f"standalone standard-tier token (batch-1, 2-pass): {t_solo_std:.2f} ms\n")

    # ---- 1. mix sweep at fixed M = 4 and M = 8 ----
    for M in (4, 8):
        print(f"## Mix sweep, M = {M} (k premium 3-pass rows, M-k standard 2-pass rows)\n")
        print("| k premium | tiered D (ms) | all-premium C (ms) | D/C | rider marginal vs standalone |")
        print("|---:|---:|---:|---:|---:|")
        # all-premium reference for this M
        def all_prem(o, i, qw, M=M):
            rx = rx_for(qw, i, M)
            return lambda: gemlite_splitk_nonuniform(rx, qw)
        C = stack_time(layers, all_prem)
        for k in range(0, M + 1):
            tiers = tuple([3] * k + [2] * (M - k))
            def tiered(o, i, qw, tiers=tiers):
                rx = rx_for(qw, i, len(tiers))
                return lambda: gemlite_splitk_nonuniform_tiered(rx, qw, tiers)
            D = stack_time(layers, tiered)
            if k == 0 or k == M:
                marg = "-"
            else:
                # premium-only batch of k rows
                def prem_only(o, i, qw, k=k):
                    rx = rx_for(qw, i, k)
                    return lambda: gemlite_splitk_nonuniform(rx, qw)
                B = stack_time(layers, prem_only)
                riders_standalone = (M - k) * t_solo_std
                marg = f"{(D - B):.2f} ms vs {riders_standalone:.1f} ms ({(D - B) / riders_standalone:.1%})"
            print(f"| {k} | {D:.2f} | {C:.2f} | {D / C:.3f} | {marg} |")
        print()

    # ---- 2. batch-size sweep at 50/50 mix: where does tiering beat all-premium materially? ----
    print("## Batch-size sweep, 50/50 premium/standard mix\n")
    print("| M | tiered D (ms) | all-premium C (ms) | saving (C-D)/C | per-row D (ms) |")
    print("|---:|---:|---:|---:|---:|")
    for M in (4, 8, 16, 32, 64, 128):
        tiers = tuple([3] * (M // 2) + [2] * (M // 2))
        def tiered(o, i, qw, tiers=tiers):
            rx = rx_for(qw, i, len(tiers))
            return lambda: gemlite_splitk_nonuniform_tiered(rx, qw, tiers)
        def all_prem(o, i, qw, M=M):
            rx = rx_for(qw, i, M)
            return lambda: gemlite_splitk_nonuniform(rx, qw)
        D = stack_time(layers, tiered)
        C = stack_time(layers, all_prem)
        print(f"| {M} | {D:.2f} | {C:.2f} | {(C - D) / C:.1%} | {D / M:.2f} |")
    print()
    print("Reading: within the memory-bound window the all-premium strategy is nearly")
    print("as cheap as tiering (the honest null); the saving grows only as the batch")
    print("turns compute-bound and the skipped third-pass arithmetic becomes real money.")


if __name__ == "__main__":
    main()
