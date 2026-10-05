"""Decode-regime speedups of fused progressive 2-bit on REAL 7B layer shapes.

The paper claims the fused 2-bit mat-vec beats dense fp16 in the
weight-bandwidth-bound regime (batch-one decode, large d) and that ApexQuant's
inference path cannot: its Algorithm 2 reconstructs the floating-point weight
and multiplies, so it executes at fp16 speed and fp16 memory regardless of the
stored bit-width. This measures batch-one latency per layer on the exact linear
shapes of the benchmarked models:

    fp16 dense GEMV (cuBLAS)     -- FP16 baseline == ApexQuant's execution path
    fused 2-bit, P = 1 / 2 / 3   -- ours (packed indices, weight never in fp)

plus a per-token estimate for a full Mistral-7B decoder stack (sum of its
linear GEMVs x 32 blocks; attention/KV and norms excluded, stated as such).
Weights are random (latency depends on shape, not values).

    uv run python experiments/decode_speedup.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from apexquant import ApexQuantConfig, turboquantize_weight
from apexquant.kernels.fused_dequant_matmul import fused_dequant_matmul

DEV = torch.device("cuda")

# (model, layer, out_features, in_features) -- real shapes from the checkpoints.
SHAPES = [
    ("Clay v1.5",   "fc1  4096x1024",    4096, 1024),
    ("TinyLlama",   "gate 5632x2048",    5632, 2048),
    ("Phi-1.5",     "fc1  8192x2048",    8192, 2048),
    ("Mistral-7B",  "qkv  6144x4096",    6144, 4096),
    ("Mistral-7B",  "up   14336x4096",  14336, 4096),
    ("Mistral-7B",  "down 4096x14336",   4096, 14336),
    ("Qwen2.5-7B",  "up   18944x3584",  18944, 3584),
    ("Qwen2.5-7B",  "down 3584x18944",   3584, 18944),
]
# Mistral-7B block GEMVs for the per-token estimate: q+k+v (as one 6144), o,
# gate, up, down; x32 blocks.
MISTRAL_BLOCK = [(6144, 4096), (4096, 4096), (14336, 4096), (14336, 4096),
                 (4096, 14336)]
DEPTH = 32


def bench(fn, iters=300, warmup=50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6  # us


def graphed(fn):
    """Capture ``fn`` into a CUDA graph; return the replay callable.

    Graph replay is the deployment mode for batch-one decode (serving stacks
    capture the whole step): it erases launch and host dispatch overhead, which
    is what dominates small layers."""
    fn()  # warm outside capture (triton compile, cache fills)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


def times_for(out_f: int, in_f: int) -> dict:
    torch.manual_seed(0)
    w = torch.randn(out_f, in_f, device=DEV)
    w16 = w.half()
    x16 = torch.randn(1, in_f, device=DEV, dtype=torch.float16)
    x32 = x16.float()
    res = {"fp16": bench(lambda: F.linear(x16, w16)),
           "fp16_g": bench(graphed(lambda: F.linear(x16, w16)))}
    for p in (1, 2, 3):
        qw = turboquantize_weight(w, ApexQuantConfig(bits=2, passes=p,
                                                      group_size=512,
                                                      codebook="apex"))
        res[f"p{p}"] = bench(lambda: fused_dequant_matmul(x32, qw))
        res[f"p{p}_g"] = bench(graphed(lambda: fused_dequant_matmul(x32, qw)))
        del qw
    return res


def main():
    torch.set_grad_enabled(False)
    name = torch.cuda.get_device_name(0)
    print(f"# Batch-one decode latency per layer -- fused 2-bit vs fp16 dense ({name})\n")
    print("# fp16 dense == ApexQuant's execution path (Algorithm 2 reconstructs fp weights).\n")
    print("| model | layer | fp16 | x1 | x2 | x2 spd | fp16-graph | x1-graph | x2-graph | x2-graph spd |")
    print("|-------|-------|-----:|---:|---:|-------:|-----------:|---------:|---------:|-------------:|")
    for model, layer, out_f, in_f in SHAPES:
        r = times_for(out_f, in_f)
        print(f"| {model} | {layer} | {r['fp16']:>5.1f} | {r['p1']:>5.1f} "
              f"| {r['p2']:>5.1f} | {r['fp16']/r['p2']:>5.2f}x "
              f"| {r['fp16_g']:>5.1f} | {r['p1_g']:>5.1f} | {r['p2_g']:>5.1f} "
              f"| {r['fp16_g']/r['p2_g']:>5.2f}x |",
              flush=True)

    print("\n## Per-token weight-GEMV estimate, full Mistral-7B decoder "
          f"(x{DEPTH} blocks; attention/KV/norms excluded)\n")
    keys = ("fp16", "p1", "p2", "p3", "fp16_g", "p1_g", "p2_g", "p3_g")
    tot = {k: 0.0 for k in keys}
    for out_f, in_f in MISTRAL_BLOCK:
        r = times_for(out_f, in_f)
        for k in keys:
            tot[k] += r[k] * DEPTH
    print("| path | eager (ms) | vs fp16 | graphed (ms) | vs fp16-graph |")
    print("|------|-----------:|--------:|-------------:|--------------:|")
    for k, label in [("fp16", "fp16 dense (== ApexQuant exec)"),
                     ("p1", "fused 2-bit x1"),
                     ("p2", "fused 2-bit x2"),
                     ("p3", "fused 2-bit x3")]:
        print(f"| {label} | {tot[k]/1000:>9.2f} | {tot['fp16']/tot[k]:>6.2f}x "
              f"| {tot[k + '_g']/1000:>11.2f} | {tot['fp16_g']/tot[k + '_g']:>12.2f}x |")


if __name__ == "__main__":
    main()
