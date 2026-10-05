"""Measure fused dequant-matmul throughput on whatever backend is active."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from ..config import ApexQuantConfig
from ..kernels.fused_dequant_matmul import fused_dequant_matmul
from ..quantize.turboquantize_weight import turboquantize_weight


@dataclass
class ThroughputResult:
    """Timing of one fused matmul configuration."""

    backend: str
    latency_ms: float
    tokens_per_s: float
    effective_bf16_gbps: float  # bandwidth a dense bf16 kernel would have needed
    compression: float


def benchmark_throughput(out_features: int, in_features: int,
                         config: ApexQuantConfig | None = None,
                         batch: int = 1, iters: int = 50, warmup: int = 10,
                         backend: str | None = None) -> ThroughputResult:
    """Time ``fused_dequant_matmul`` and report tokens/s and effective bandwidth.

    Runs on whatever backend the dispatcher selects (torch on CPU here; Triton or
    CuTile on a GPU). ``effective_bf16_gbps`` is the HBM bandwidth a dense bf16
    GEMM would have needed to hit the same latency -- a hardware-independent way
    to see the memory win, since the fused kernel achieves it while moving far
    fewer bytes.

    Args:
        out_features, in_features: layer shape.
        config: quantization config (defaults to 4-bit single pass).
        batch: tokens per call.
        iters, warmup: timed and untimed iterations.
        backend: force ``"torch"`` / ``"triton"`` / ``"cutile"``.

    Returns:
        A :class:`ThroughputResult`.
    """
    config = config or ApexQuantConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight = torch.randn(out_features, in_features, device=device)
    qw = turboquantize_weight(weight, config)
    x = torch.randn(batch, in_features, device=device)

    def run() -> None:
        fused_dequant_matmul(x, qw, force_backend=backend)
        if device == "cuda":
            torch.cuda.synchronize()

    for _ in range(warmup):
        run()
    start = time.perf_counter()
    for _ in range(iters):
        run()
    elapsed = time.perf_counter() - start

    latency_ms = elapsed / iters * 1e3
    tokens_per_s = batch * iters / elapsed
    bf16_bytes = out_features * in_features * 2
    effective_bf16_gbps = bf16_bytes * iters / elapsed / 1e9
    compression = bf16_bytes / qw.nbytes()

    resolved = backend or ("cuda-kernel" if device == "cuda" else "torch")
    return ThroughputResult(
        backend=resolved,
        latency_ms=latency_ms,
        tokens_per_s=tokens_per_s,
        effective_bf16_gbps=effective_bf16_gbps,
        compression=compression,
    )
