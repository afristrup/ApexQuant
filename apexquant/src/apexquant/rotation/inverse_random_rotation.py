"""Invert the randomized Hadamard rotation along the last dimension."""

from __future__ import annotations

import os

import torch

from .hadamard_backend import matmul_rotation_max_dim
from .group_signs import group_signs
from .random_sign_vector import random_sign_vector
from .rotation_matrix import MATMUL_ROTATION_MAX_DIM, rotation_matrix
from .walsh_hadamard_transform import walsh_hadamard_transform


def inverse_random_rotation(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Undo :func:`random_rotation`, i.e. apply ``R^T = diag(sign) @ WHT``.

    Because the normalized WHT is symmetric and its own inverse, and the sign
    diagonal is its own inverse, the transpose is just "WHT then re-sign":
    ``R^T(y) = sign * WHT(y)``. Used to reconstruct dequantized weights. For small
    groups this mirrors :func:`random_rotation` and applies the cached rotation
    matrix's transpose as a single matmul.

    Args:
        x: Tensor whose last dim is a power of two.
        seed: The same seed passed to :func:`random_rotation`.

    Returns:
        Inverse-rotated tensor of the same shape.
    """
    d = x.shape[-1]
    if os.environ.get("APEXQUANT_GROUP_SIGNS") == "1" and x.ndim >= 2:
        # One independent sign vector per group (axis -2) instead of one per pass.
        signs = group_signs(x.shape[-2], d, int(seed), str(x.device)).to(x.dtype)
        return walsh_hadamard_transform(x) * signs
    if d <= matmul_rotation_max_dim(x, MATMUL_ROTATION_MAX_DIM):
        return x @ rotation_matrix(d, int(seed), str(x.device)).to(x.dtype).transpose(-2, -1)
    signs = random_sign_vector(d, seed, device=x.device).to(x.dtype)
    return walsh_hadamard_transform(x) * signs
