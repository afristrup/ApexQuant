"""Construct a Clay encoder of one of the published sizes."""

from __future__ import annotations

from ..constants import DEFAULT_PATCH_SIZE
from .encoder import Encoder
from .sizes import SIZES


def build_encoder(
    size: str,
    mask_ratio: float = 0.0,
    patch_size: int = DEFAULT_PATCH_SIZE,
    shuffle: bool = False,
) -> Encoder:
    """Build the encoder for ``size``, defaulting to inference settings.

    Args:
        size: One of ``tiny``, ``small``, ``base``, ``large``.
        mask_ratio: Fraction of patches to withhold; 0 keeps the whole grid.
        patch_size: Side of a square patch in pixels.
        shuffle: Randomise which patches are withheld.

    Returns:
        An untrained :class:`~clay.model.encoder.Encoder`.
    """
    if size not in SIZES:
        raise ValueError(f"unknown model size {size!r}, expected one of {list(SIZES)}")
    return Encoder(
        mask_ratio=mask_ratio,
        patch_size=patch_size,
        shuffle=shuffle,
        **SIZES[size]["encoder"],
    )
