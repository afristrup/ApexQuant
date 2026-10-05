"""Rows per program for the fused Walsh-Hadamard rotation."""

from __future__ import annotations

# A tile is BLOCK_R x D floats and every butterfly stage permutes the whole
# thing, so the tile has to stay inside the register budget or the stages spill.
# Holding BLOCK_R * D near constant is what that asks for, and it is worth
# getting right: measured inside the Clay forward, g=512 at 32 rows costs
# 1.065 ms a forward against 0.527 at 4 rows, and g=128 wants 8. Wider is always
# worse here, 64 rows being 0.580 ms at g=128 and 1.974 at g=512.
TILE_ELEMENTS = 1024
MIN_ROW_TILE = 4


def wht_row_tile(dim: int) -> int:
    """Rows per program for a group of ``dim`` coordinates.

    Args:
        dim: Group size.

    Returns:
        The tile height to launch with.
    """
    return max(MIN_ROW_TILE, TILE_ELEMENTS // dim)
