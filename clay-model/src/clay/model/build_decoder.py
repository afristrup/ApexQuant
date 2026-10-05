"""Construct a Clay decoder of one of the published sizes."""

from __future__ import annotations

from ..constants import DEFAULT_PATCH_SIZE
from .decoder import Decoder
from .sizes import SIZES


def build_decoder(
    size: str, mask_ratio: float = 0.0, patch_size: int = DEFAULT_PATCH_SIZE
) -> Decoder:
    """Build the decoder for ``size``, wired to that size's encoder width.

    Args:
        size: One of ``tiny``, ``small``, ``base``, ``large``.
        mask_ratio: Fraction of patches the encoder withholds.
        patch_size: Side of a square patch in pixels.

    Returns:
        An untrained :class:`~clay.model.decoder.Decoder`.
    """
    if size not in SIZES:
        raise ValueError(f"unknown model size {size!r}, expected one of {list(SIZES)}")
    return Decoder(
        mask_ratio=mask_ratio,
        patch_size=patch_size,
        encoder_dim=SIZES[size]["encoder"]["dim"],
        **SIZES[size]["decoder"],
    )
