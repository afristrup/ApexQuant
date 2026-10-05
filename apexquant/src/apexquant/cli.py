"""Command-line entry point: ``apexquant <command>``."""

from __future__ import annotations

import argparse

import torch

from .codebook.apex_beta_levels import apex_beta_levels
from .config import ApexQuantConfig
from .quantize.dequantize_from_indices import dequantize_from_indices
from .quantize.dequantize_weight import dequantize_weight
from .quantize.quantize_to_indices import quantize_to_indices
from .quantize.turboquantize_weight import turboquantize_weight
from .vq.e8p_codec import E8PCodec
from .vq.trellis_codec import TrellisCodec

# Table 2's operating point: d=512, b=2 bits/coordinate, L=14 trellis states.
STAGE_GROUP_SIZE = 512
STAGE_BITS = 2
STAGE_TRELLIS_BITS = 14
DEFAULT_STAGE_SAMPLES = 512
DEFAULT_STAGE_SEED = 0

# Table 2's published rho^2 at the operating point above.
PAPER_RHO2 = {
    "scalar": 0.1171,
    "e8": 0.0906,
    "trellis": 0.0720,
}


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run a subcommand.

    Subcommands:
        ``stages``    Table 2 sanity check: measured vs. published rho^2 for
                      the scalar, E8, and trellis stages.
        ``benchmark`` Report reconstruction error of turboquantizing a random
                      matrix at the chosen bit budget.
        ``throughput`` Time the fused kernel and print the roofline model.

    Returns process exit code.
    """
    parser = argparse.ArgumentParser(prog="apexquant", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_stages = sub.add_parser("stages", help="Table 2 sanity check: measured vs. paper rho^2")
    p_stages.add_argument("--samples", type=int, default=DEFAULT_STAGE_SAMPLES,
                          help="unit-sphere probe vectors per stage")
    p_stages.add_argument("--seed", type=int, default=DEFAULT_STAGE_SEED)

    p_bench = sub.add_parser("benchmark", help="reconstruction error on a random matrix")
    p_bench.add_argument("--out", type=int, default=1024)
    p_bench.add_argument("--in", dest="in_features", type=int, default=1024)
    p_bench.add_argument("--passes", type=int, default=1)
    p_bench.add_argument("--group-size", type=int, default=128)
    p_bench.add_argument("--bits", type=int, default=4, choices=[2, 4, 8])
    p_bench.add_argument("--codebook", default="gaussian", choices=["gaussian", "apex", "uniform"])

    p_tput = sub.add_parser("throughput", help="time the fused kernel + roofline model")
    p_tput.add_argument("--out", type=int, default=4096)
    p_tput.add_argument("--in", dest="in_features", type=int, default=4096)
    p_tput.add_argument("--bits", type=int, default=4, choices=[2, 4, 8])
    p_tput.add_argument("--passes", type=int, default=1)
    p_tput.add_argument("--batch", type=int, default=1)

    args = parser.parse_args(argv)
    if args.command == "stages":
        return _run_stages(args)
    if args.command == "benchmark":
        return _run_benchmark(args)
    if args.command == "throughput":
        return _run_throughput(args)
    return 1


def _sample_sphere(n: int, dim: int, seed: int) -> torch.Tensor:
    """``n`` points drawn uniformly from the unit sphere ``S^{dim-1}``."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=generator)
    return x / x.norm(dim=1, keepdim=True)


def _rho_squared(samples: torch.Tensor, recon: torch.Tensor) -> float:
    """Mean squared reconstruction error of unit-norm ``samples``, i.e. rho^2."""
    return (recon - samples).pow(2).sum(dim=1).mean().item()


def _run_stages(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    samples = _sample_sphere(args.samples, STAGE_GROUP_SIZE, args.seed)

    levels = apex_beta_levels(STAGE_BITS, STAGE_GROUP_SIZE)
    scalar_recon = dequantize_from_indices(quantize_to_indices(samples, levels), levels)
    measured = {"scalar": _rho_squared(samples, scalar_recon)}

    e8_codec = E8PCodec(STAGE_GROUP_SIZE, device="cpu")
    measured["e8"] = _rho_squared(samples, e8_codec(samples))

    trellis_codec = TrellisCodec(STAGE_GROUP_SIZE, bits=STAGE_BITS,
                                 trellis_bits=STAGE_TRELLIS_BITS, device="cpu")
    measured["trellis"] = _rho_squared(samples, trellis_codec(samples))

    print(f"Table 2 sanity check: d={STAGE_GROUP_SIZE}  b={STAGE_BITS}  "
          f"L={STAGE_TRELLIS_BITS}  samples={args.samples}")
    print(f"{'stage':10s}{'measured rho^2':>16s}{'paper rho^2':>14s}")
    for name in ("scalar", "e8", "trellis"):
        print(f"{name:10s}{measured[name]:16.4f}{PAPER_RHO2[name]:14.4f}")
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    torch.manual_seed(0)
    weight = torch.randn(args.out, args.in_features)
    config = ApexQuantConfig(bits=args.bits, passes=args.passes,
                             group_size=args.group_size, codebook=args.codebook)
    qw = turboquantize_weight(weight, config)
    recon = dequantize_weight(qw)

    rel = (recon - weight).norm() / weight.norm()
    dense_bytes = weight.numel() * weight.element_size()
    print(f"weight {tuple(weight.shape)}  bits={args.bits}  passes={args.passes}  "
          f"group_size={args.group_size}  codebook={args.codebook}")
    print(f"relative reconstruction error: {rel.item():.4f}")
    print(f"compression: {dense_bytes / qw.nbytes():.2f}x smaller than fp32")
    return 0


def _run_throughput(args: argparse.Namespace) -> int:
    from .throughput.benchmark_throughput import benchmark_throughput
    from .throughput.memory_traffic_model import memory_traffic_model

    config = ApexQuantConfig(bits=args.bits, passes=args.passes)
    model = memory_traffic_model(args.out, args.in_features, args.bits,
                                 passes=args.passes, batch=args.batch)
    result = benchmark_throughput(args.out, args.in_features, config, batch=args.batch)

    print(f"layer {args.out}x{args.in_features}  bits={args.bits}  passes={args.passes}  batch={args.batch}")
    print(f"roofline: {model.traffic_ratio:.2f}x less weight traffic than bf16 "
          f"({'memory-bound' if model.memory_bound else 'compute-bound'}, "
          f"AI={model.arithmetic_intensity:.1f} FLOP/byte)")
    print(f"measured backend={result.backend}: {result.latency_ms:.3f} ms/call, "
          f"{result.tokens_per_s:.0f} tokens/s, {result.compression:.2f}x smaller")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
