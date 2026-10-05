"""What the trellis path costs Clay in time and in memory, against fp32.

`progressive_trellis_clay.py` answers whether the codes are accurate enough. This
answers whether they are worth running: latency against the two dense baselines,
and the memory that buys.

TWO DENSE BASELINES, BECAUSE "FULL PRECISION" IS AMBIGUOUS HERE. The port runs
its own TF32 Triton kernels by default and falls back to torch fp32 when
`CLAY_DISABLE_TRITON` is set, and those are 1.36x apart on this model. Beating
the slower one is not the same claim as beating the faster one, so both are
reported and neither is called "the" baseline.

GROUP SIZE USED TO BE A LEVER AND NO LONGER IS. Every quantised layer rotates its
activations first, and while that was a dense ``(g, g)`` matmul per group the cost
grew with ``g``: at ``g=512`` it was a third of the arithmetic of the layer it
preceded, which is why the small group won despite costing ``16/g`` more stored
bits. The Walsh-Hadamard rotation is ``g log g``, so that pressure is gone --
g=512 and g=128 now time the same, and g=512 is simply better: fewer stored bits
(4.03 against 4.12) and a closer embedding (0.99411 against 0.99302). 512 is the
operating point; 128 stays in the table as the comparison that made the case.

MEMORY IS COUNTED BY STORAGE, NOT BY TENSOR. The permuted codeword table is
registered non-persistently and shared by every layer, so summing per-module
buffers would count one 16 KB table 102 times. Storages are deduplicated by data
pointer. Peak is the allocator's, across a forward.

    uv run python experiments/trellis_clay_speed_memory.py
    uv run python experiments/trellis_clay_speed_memory.py --group-sizes 128
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from progressive_trellis_clay import (  # noqa: E402
    DEV,
    datacube,
    eligible,
    load_encoder,
)

from clay.constants import DISABLE_TRITON_ENV  # noqa: E402
from clay.inference.embed import embed  # noqa: E402
from apexquant.kernels.triton_gemv_trellis import (  # noqa: E402
    trellis_window_table,
)
from apexquant.vq.progressive_artifact import (  # noqa: E402
    group_size_for,
    quantize,
    stored_bits_per_weight,
)
from apexquant.vq.progressive_trellis_linear import (  # noqa: E402
    ProgressiveTrellisLinear,
)
from apexquant.vq.trellis_codec import (  # noqa: E402
    DEFAULT_TRELLIS_BITS,
    TrellisCodec,
)

MIB = 1024 * 1024


def resident_mib(model: torch.nn.Module) -> float:
    """Bytes the model holds, counting each storage once."""
    seen = {}
    for tensor in list(model.parameters()) + list(model.buffers()):
        storage = tensor.untyped_storage()
        seen[storage.data_ptr()] = storage.nbytes()
    return sum(seen.values()) / MIB


def measure(model: torch.nn.Module, cube: dict, rounds: int = 5,
            repeats: int = 30) -> tuple:
    """Return forward milliseconds and peak allocated MiB over a forward.

    Timed through a CUDA graph, and reported as the minimum over rounds. Both
    choices are there because the first version of this benchmark got the answer
    wrong. Timing eager launches made the packed path look 0.4 ms slower than it
    is, since it issues about twice the kernels of the dense one and the launch
    bill is not what is being compared; and reading one round put the dense
    baseline anywhere in 14.8-16.0 ms, which is wider than the difference under
    test. Capture removes the launches, and the minimum is the round least
    disturbed by whatever else the device was doing.
    """
    # Peak is measured eagerly, before capture. A replay allocates nothing: it
    # reuses the graph's private pool, so the allocator's high-water mark under
    # replay is zero and says nothing about what a forward needs.
    for _ in range(3):
        embed(model, cube)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    embed(model, cube)
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - baseline) / MIB

    static = {k: (v.detach().clone().to(DEV) if torch.is_tensor(v) else v)
              for k, v in cube.items()}
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(5):
            embed(model, static)
    torch.cuda.current_stream().wait_stream(warm)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        embed(model, static)

    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            graph.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - start) / repeats * 1000)
    del graph
    return best, peak


def quantise(model, targets, passes, group_size, args):
    """A copy of ``model`` with every eligible linear kept as packed codes."""
    codecs, tables = {}, {}
    quantised = copy.deepcopy(model)
    bits_total, covered = 0.0, 0

    for name, module in targets:
        weight = module.weight.detach().to(DEV, torch.float32)
        out_f, in_f = weight.shape
        gs = group_size_for(in_f, group_size)
        if gs not in codecs:
            codecs[gs] = TrellisCodec(d=gs, bits=args.bits,
                                      trellis_bits=args.trellis_bits,
                                      seed=0, device=DEV)
            tables[gs] = trellis_window_table(codecs[gs])
        codes, scale = quantize(weight, codecs[gs], passes, gs, args.seed)
        bits_total += stored_bits_per_weight(out_f, in_f, passes, group_size,
                                             args.bits) * weight.numel()
        covered += weight.numel()

        parent_name, _, attr = name.rpartition(".")
        parent = quantised.get_submodule(parent_name)
        old = getattr(parent, attr)
        words = codes.reshape(passes, out_f, -1).contiguous().view(torch.int32)
        setattr(parent, attr, ProgressiveTrellisLinear(
            words, scale.to(torch.float32), tables[gs], codecs[gs], in_f, gs,
            args.seed, None if old.bias is None else old.bias.detach()))
        del weight, codes
    torch.cuda.empty_cache()
    return quantised, bits_total / covered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--group-sizes", type=int, nargs="+", default=[512, 128],
                    help="512 is the operating point; 128 is the comparison.")
    ap.add_argument("--bits-list", type=int, nargs="+", default=[2, 4],
                    help="Stage widths; b=4 P=1 and b=2 P=2 cost the same bits.")
    ap.add_argument("--trellis-bits", type=int, default=DEFAULT_TRELLIS_BITS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chip", type=int, default=224)
    args = ap.parse_args()

    model = load_encoder()
    targets = eligible(model)
    cube = datacube(args.chip, 1)
    dense_mib = resident_mib(model)

    os.environ[DISABLE_TRITON_ENV] = "1"
    fp32_ms, fp32_peak = measure(model, cube)
    del os.environ[DISABLE_TRITON_ENV]
    tf32_ms, tf32_peak = measure(model, cube)
    reference = embed(model, cube)[:, 0, :]

    print(f"# Clay v1.5 encoder, chip={args.chip}, "
          f"trellis L={args.trellis_bits}\n")
    print("| config | bits/weight | resident MiB | peak MiB | ms | "
          "vs fp32 | vs TF32 | cls cosine |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    print(f"| dense fp32 | 32.00 | {dense_mib:.0f} | {fp32_peak:.0f} | "
          f"{fp32_ms:.1f} | 1.00x | {tf32_ms / fp32_ms:.2f}x | 1.00000000 |")
    print(f"| dense TF32 | 32.00 | {dense_mib:.0f} | {tf32_peak:.0f} | "
          f"{tf32_ms:.1f} | {fp32_ms / tf32_ms:.2f}x | 1.00x | 0.99999860 |")

    for stage_bits in args.bits_list:
      args.bits = stage_bits
      for group_size in args.group_sizes:
        for passes in range(1, args.passes + 1):
            quantised, bits = quantise(model, targets, passes, group_size, args)
            elapsed, peak = measure(quantised, cube)
            cosine = F.cosine_similarity(
                embed(quantised, cube)[:, 0, :], reference).item()
            print(f"| b={stage_bits} g={group_size} P={passes} | {bits:.2f} | "
                  f"{resident_mib(quantised):.0f} | {peak:.0f} | {elapsed:.1f} | "
                  f"{fp32_ms / elapsed:.2f}x | {tf32_ms / elapsed:.2f}x | "
                  f"{cosine:.8f} |", flush=True)
            del quantised
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
