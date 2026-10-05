"""Run the encoder over a datacube and return its tokens."""

from __future__ import annotations

import torch

from ..model.encoder import Encoder


def embed(encoder: Encoder, datacube: dict[str, torch.Tensor]) -> torch.Tensor:
    """Encode a datacube into tokens, with no patch withheld.

    Embedding is only meaningful over the whole grid, so an encoder still
    configured for masked pretraining is rejected rather than quietly returning
    a partial view. Token 0 is the class token, the usual chip-level embedding;
    tokens 1 onward are the patches in grid order.

    Args:
        encoder: An encoder with ``mask_ratio`` 0 and ``shuffle`` off.
        datacube: ``pixels`` ``(B, C, H, W)``, ``time`` ``(B, 4)``, ``latlon``
            ``(B, 4)``, scalar ``gsd``, and ``waves`` ``(C,)``.

    Returns:
        Tokens ``(B, 1 + num_patches, dim)``.
    """
    if encoder.mask_ratio != 0:
        raise ValueError(f"mask_ratio must be 0 to embed, got {encoder.mask_ratio}")
    if encoder.shuffle:
        raise ValueError("shuffle must be off to embed")

    with torch.no_grad():
        tokens, _, _, _ = encoder(datacube)
    return tokens
