"""Fast, orthonormal Walsh-Hadamard transform along the last dimension."""

from __future__ import annotations

import math

import torch

from .hadamard_backend import _load_cuda_kernel, use_cuda_hadamard


def walsh_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Apply the normalized Walsh-Hadamard transform over the last axis.

    Scales by ``1/sqrt(d)`` so the transform is orthonormal. Being orthonormal
    and symmetric, it is its own inverse: ``wht(wht(x)) == x``.

    Two backends compute this, selected by :func:`use_cuda_hadamard`: a single
    fused CUDA kernel when one is installed and applicable, otherwise a portable
    in-place butterfly (``O(d log d)`` rather than the ``O(d^2)`` dense matmul).
    The butterfly issues ``log2(d)`` separate torch ops, which on GPU costs
    ``log2(d)`` kernel launches; the CUDA kernel does the whole thing in one.
    They agree to floating-point rounding.

    Args:
        x: Tensor whose last dimension ``d`` is a power of two.

    Returns:
        Transformed tensor of the same shape.
    """
    d = x.shape[-1]
    if d & (d - 1) != 0:
        raise ValueError(f"last dim must be a power of two, got {d}")

    if use_cuda_hadamard(x):
        # The kernel wants the transformed axis contiguous, and folds the
        # normalisation in for free via its `scale` argument.
        hadamard_transform = _load_cuda_kernel()
        return hadamard_transform(x.contiguous(), scale=1.0 / math.sqrt(d))

    y = x.clone()
    h = 1
    while h < d:
        # View the last axis as (..., blocks, 2, h) and butterfly the pair.
        y = y.view(*y.shape[:-1], d // (2 * h), 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = torch.stack([a + b, a - b], dim=-2)
        y = y.reshape(*y.shape[:-3], d)
        h *= 2

    return y / math.sqrt(d)
