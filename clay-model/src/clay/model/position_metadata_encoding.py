"""The per-patch encoding: GSD-scaled position, then time and location."""

from __future__ import annotations

import torch

from ..constants import METADATA_DIM
from ..ops.sincos_2d_with_gsd import sincos_2d_with_gsd
from .patch_grid_size import patch_grid_size


def position_metadata_encoding(  # noqa: PLR0913
    num_patches: int,
    dim: int,
    gsd: torch.Tensor,
    time: torch.Tensor,
    latlon: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build the ``(B, num_patches, dim)`` tensor added to every patch token.

    The last :data:`~clay.constants.METADATA_DIM` channels are not sinusoids at
    all: they carry the normalised week, hour, latitude and longitude verbatim,
    repeated across patches. Encoder and decoder must build this identically, or
    the decoder scatters reconstructions to the wrong places, so both call here.

    Args:
        num_patches: Tokens per sample, a square number.
        dim: Token width; the sinusoid table gets ``dim - METADATA_DIM`` of it.
        gsd: Scalar ground sample distance in metres.
        time: Normalised week and hour, ``(B, 4)``.
        latlon: Normalised latitude and longitude, ``(B, 4)``.
        device: Device to build on.

    Returns:
        Tensor of shape ``(B, num_patches, dim)``, detached from the graph.
    """
    grid = patch_grid_size(num_patches)
    table = sincos_2d_with_gsd(grid, grid, dim - METADATA_DIM, gsd, device).detach()
    metadata = torch.hstack((time, latlon)).to(device).detach()

    batch = metadata.shape[0]
    table = table.unsqueeze(0).expand(batch, -1, -1)
    metadata = metadata.unsqueeze(1).expand(-1, num_patches, -1)
    return torch.cat((table, metadata), dim=-1)
