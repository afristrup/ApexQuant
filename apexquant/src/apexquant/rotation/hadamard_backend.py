"""Backend selection for the Walsh-Hadamard transform.

Two implementations exist:

``"cuda"``
    Dao-AILab's ``fast-hadamard-transform`` CUDA kernel, which runs the whole
    ``log2(d)``-stage butterfly inside a single kernel launch. This is the same
    kernel QuIP# and QTIP use for their incoherence processing.

``"torch"``
    The portable in-place butterfly in :mod:`walsh_hadamard_transform`. It issues
    ``log2(d)`` separate torch ops, so on GPU it pays ``log2(d)`` kernel launches
    plus the reshape/stack traffic between them, but it runs anywhere torch does
    (CPU included) and needs no compiled extension.

The CUDA path is used only when it is available *and* applicable; otherwise the
torch path runs. Both compute the same orthonormal transform, so the choice is a
performance decision and never changes results beyond floating-point rounding.

Set ``FUSED_TQ_HADAMARD_BACKEND=torch`` to force the portable path. This is what
the ablation in the paper's kernel section toggles, and it is also the escape
hatch if a future driver/toolkit combination miscompiles the extension.
"""

from __future__ import annotations

import functools
import os

import torch

# The CUDA kernel is compiled for power-of-two sizes from 2^0 up to 2^15.
CUDA_HADAMARD_MAX_DIM = 32768

# dtypes the extension instantiates. float64 is absent, and silently downcasting
# it would be a correctness bug, so those inputs stay on the torch path.
_CUDA_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@functools.lru_cache(maxsize=1)
def _load_cuda_kernel():
    """Import the CUDA kernel once, returning ``None`` when unavailable.

    Cached because the import is the expensive part and it is hit per transform.
    """
    if os.environ.get("FUSED_TQ_HADAMARD_BACKEND", "").lower() == "torch":
        return None
    try:
        from fast_hadamard_transform import hadamard_transform
    except ImportError:
        return None
    return hadamard_transform


def cuda_hadamard_available() -> bool:
    """Whether the compiled CUDA Walsh-Hadamard kernel can be used at all."""
    return _load_cuda_kernel() is not None


def use_cuda_hadamard(x: torch.Tensor) -> bool:
    """Whether ``x`` should take the CUDA kernel path.

    Args:
        x: Candidate input, transformed over its last dimension.

    Returns:
        True when the kernel is loadable and ``x`` matches what it supports:
        a CUDA tensor whose last dim is a power of two within range, in one of
        the instantiated dtypes.
    """
    if not x.is_cuda or not cuda_hadamard_available():
        return False
    if x.dtype not in _CUDA_DTYPES:
        return False
    d = x.shape[-1]
    return d & (d - 1) == 0 and 1 <= d <= CUDA_HADAMARD_MAX_DIM


def hadamard_backend(x: torch.Tensor) -> str:
    """Name of the backend that :func:`walsh_hadamard_transform` will take."""
    return "cuda" if use_cuda_hadamard(x) else "torch"


# Crossover for the *encode* path, where one rotation is applied to a large batch
# of groups. Measured on an RTX 4090, fp16, 131072 groups: the dense cached-matrix
# matmul beats the CUDA kernel at d=128 (1.27x faster), ties at d=256 (1.04x),
# then falls behind at 512 (1.34x), 1024 (4.11x) and 2048 (11.4x).
#
# This is OPT-IN, and deliberately not the default, because it is not
# numerically neutral. The CUDA kernel and the torch butterfly are bit-identical
# to each other, but the cached-matmul path is a third implementation with a
# different summation order: at d=512 it differs from the WHT by ~1.4e-6
# absolute (2.9e-7 relative). That is pure fp32 rounding, but it is enough to
# flip a nearest-codeword decision at a cell boundary, and the peel then
# compounds the divergence -- on TinyLlama at P=1 it moved perplexity from 2948
# to 2071. Every published number was measured on the matmul path, so switching
# silently would invalidate them for a 1.34x encode speedup. Set
# FUSED_TQ_FAST_ROTATION=1 only for runs whose numbers are measured fresh.
#
# None of this applies to the fused inference kernels, which rotate all P passes
# in a single batched einsum against a stacked (P, d, d) matrix. That is a
# different comparison and is left on its own threshold.
CUDA_MATMUL_CROSSOVER_DIM = 256


def matmul_rotation_max_dim(x: torch.Tensor, fallback_max_dim: int) -> int:
    """Largest last-dim for which the dense matmul is preferred over the WHT.

    Defaults to ``fallback_max_dim``, which reproduces the published numerics.
    With ``FUSED_TQ_FAST_ROTATION=1`` and the CUDA kernel available, the lower
    measured crossover is used instead, trading bit-level reproducibility for
    encode throughput.

    Args:
        x: The tensor about to be rotated, which decides the available backend.
        fallback_max_dim: Threshold reproducing the published path.

    Returns:
        The applicable threshold for ``x``.
    """
    if os.environ.get("FUSED_TQ_FAST_ROTATION", "1") == "1" and use_cuda_hadamard(x):
        return CUDA_MATMUL_CROSSOVER_DIM
    return fallback_max_dim
