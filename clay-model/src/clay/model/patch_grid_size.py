"""Recover the side length of the square patch grid from a token count."""

from __future__ import annotations

import math


def patch_grid_size(num_patches: int) -> int:
    """Return ``g`` such that ``g * g == num_patches``.

    Clay only ever patches square chips, and both the position encoding and the
    decoder's scatter depend on that. A non-square token count means the caller
    fed a chip whose side is not a multiple of the patch size, which is worth a
    loud failure rather than a silently truncated grid.

    Args:
        num_patches: Number of patch tokens in one sample.

    Returns:
        The grid side length.
    """
    side = math.isqrt(num_patches)
    if side * side != num_patches:
        raise ValueError(f"{num_patches} patches do not form a square grid")
    return side
