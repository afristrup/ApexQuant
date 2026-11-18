"""Apply the randomized Hadamard rotation along the last dimension."""

from __future__ import annotations

import os

import torch

from .hadamard_backend import matmul_rotation_max_dim
from .group_signs import group_signs
from .random_sign_vector import random_sign_vector
from .rotation_matrix import MATMUL_ROTATION_MAX_DIM, rotation_matrix
from .walsh_hadamard_transform import walsh_hadamard_transform


def random_rotation(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Rotate ``x`` by ``R = WHT @ diag(sign(seed))`` over the last axis.

    Random rotation is what makes cheap per-coordinate scalar quantization
    near-optimal: it spreads any heavy-tailed / spiky weight energy evenly so
    every coordinate looks Gaussian, which is exactly what the Lloyd-Max
    codebook is tuned for. ``R`` is orthonormal, so it preserves norms and its
    inverse is ``R^T`` (see :func:`inverse_random_rotation`).

    For small group sizes the rotation is applied as a single cached-matrix
    matmul rather than the ``O(log d)`` butterfly -- one cuBLAS call instead of a
    chain of tiny kernel launches. Where that threshold sits depends on which
    Walsh-Hadamard backend is available (see :func:`matmul_rotation_max_dim`):
    with the compiled CUDA kernel installed the matmul only wins below ``d=256``,
    without it the matmul stays ahead much further. The paths are identical up to
    floating point.

    Args:
        x: Tensor whose last dim is a power of two.
        seed: Seed selecting the sign pattern.

    Returns:
        Rotated tensor of the same shape.
    """
    d = x.shape[-1]
    if os.environ.get("APEXQUANT_GROUP_SIGNS") == "1" and x.ndim >= 2:
        # One independent sign vector per group (axis -2) instead of one per pass.
        signs = group_signs(x.shape[-2], d, int(seed), str(x.device)).to(x.dtype)
        return walsh_hadamard_transform(x * signs)
    if d <= matmul_rotation_max_dim(x, MATMUL_ROTATION_MAX_DIM):
        return x @ rotation_matrix(d, int(seed), str(x.device)).to(x.dtype)
    signs = random_sign_vector(d, seed, device=x.device).to(x.dtype)
    return walsh_hadamard_transform(x * signs)
