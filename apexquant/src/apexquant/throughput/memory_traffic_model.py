"""Analytical roofline for the fused dequant-matmul (why quantizing is faster)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrafficModel:
    """Memory-traffic accounting for one quantized linear layer.

    Attributes:
        weight_bytes_bf16: bytes to stream the dense bf16 weight once.
        weight_bytes_fused: bytes to stream the packed indices + scales + codebook.
        traffic_ratio: ``bf16 / fused`` -- the memory-bound speedup ceiling.
        arithmetic_intensity: FLOPs per byte of weight traffic at this batch.
        memory_bound: whether weight traffic (not compute) sets the latency.
    """

    weight_bytes_bf16: int
    weight_bytes_fused: int
    traffic_ratio: float
    arithmetic_intensity: float
    memory_bound: bool


def memory_traffic_model(out_features: int, in_features: int, bits: int,
                         passes: int = 1, group_size: int = 128, batch: int = 1,
                         flops_per_byte_ridge: float = 100.0) -> TrafficModel:
    """Predict the speedup of the fused kernel from memory traffic alone.

    LLM/EO decode is weight-bandwidth-bound: at small batch the GEMM reads the
    whole weight once and does little compute per byte, so latency tracks *bytes
    moved*, not FLOPs. The fused kernel never materializes the float weight -- it
    streams ``bits``-bit indices instead of 16-bit values -- so the speedup
    ceiling is simply the traffic ratio. This is the honest throughput story:
    ~4x at 4 bits, ~8x at 2 bits, before any kernel tuning.

    Args:
        out_features, in_features: layer shape.
        bits: index bit-width.
        passes: residual passes (each adds one index per weight).
        group_size: coordinates per scale.
        batch: tokens processed together (decode ~= 1).
        flops_per_byte_ridge: the accelerator's compute:bandwidth ridge point
            (H100 bf16 ~ a few hundred FLOP/byte); above it a kernel is
            compute-bound, below it memory-bound.

    Returns:
        A :class:`TrafficModel`.
    """
    n = out_features * in_features
    num_groups = -(-in_features // group_size)

    weight_bytes_bf16 = n * 2
    index_bytes = n * bits // 8 * passes
    scale_bytes = out_features * num_groups * 4 * passes
    codebook_bytes = (2 ** bits) * 4
    weight_bytes_fused = index_bytes + scale_bytes + codebook_bytes

    flops = 2 * batch * n  # multiply-add per weight per token
    arithmetic_intensity = flops / max(weight_bytes_fused, 1)

    return TrafficModel(
        weight_bytes_bf16=weight_bytes_bf16,
        weight_bytes_fused=weight_bytes_fused,
        traffic_ratio=weight_bytes_bf16 / max(weight_bytes_fused, 1),
        arithmetic_intensity=arithmetic_intensity,
        memory_bound=arithmetic_intensity < flops_per_byte_ridge,
    )
