"""Precomputed randomized-Hadamard rotation matrix (fast path for small groups)."""

from __future__ import annotations

from functools import lru_cache

import torch

from .random_sign_vector import random_sign_vector
from .walsh_hadamard_transform import walsh_hadamard_transform

# Group sizes at or below this use the dense-matrix matmul instead of the *torch*
# butterfly. The matmul is one cuBLAS call vs O(log d) tiny kernel launches, so it
# wins in the launch-bound regime (5-27x on an RTX 4090 for d in 128..512); above
# this the butterfly's O(d log d) flops and the d*d matrix memory take over.
#
# This threshold is the no-CUDA-kernel fallback. When Dao-AILab's compiled
# Walsh-Hadamard kernel is installed the launch-bound argument mostly evaporates,
# because that backend is a single launch too; the encode path then switches at
# CUDA_MATMUL_CROSSOVER_DIM instead. See :func:`matmul_rotation_max_dim`.
MATMUL_ROTATION_MAX_DIM = 1024


@lru_cache(maxsize=256)
def rotation_matrix(dim: int, seed: int, device_str: str) -> torch.Tensor:
    """Return the dense ``R`` with ``x @ R == random_rotation(x, seed)``.

    ``R`` is the randomized Hadamard rotation applied to the identity: row ``i``
    is ``random_rotation(e_i)``. Because the rotation is linear, applying it to a
    batch is then a single ``x @ R`` matmul. Built once via the butterfly and
    cached per ``(dim, seed, device)``; the result is read-only.

    Args:
        dim: Group size (a power of two).
        seed: Rotation seed.
        device_str: Device string for the cache key and the returned tensor.

    Returns:
        Float32 ``(dim, dim)`` orthonormal rotation matrix.
    """
    signs = random_sign_vector(dim, seed, device=device_str)
    return walsh_hadamard_transform(torch.eye(dim, device=device_str) * signs)
