"""The Clay decoder: refill the masked positions and project back to pixels."""

from __future__ import annotations

import torch
from torch import nn

from ..constants import NUM_LATENT_TOKENS, WAVE_DIM
from .dynamic_embedding import DynamicEmbedding
from .position_metadata_encoding import position_metadata_encoding
from .transformer import Transformer


class Decoder(nn.Module):
    """Reconstruct every patch from the encoder's surviving tokens.

    Masked positions are seeded with a single learned token, so the only thing
    telling them apart is the position and metadata encoding added on top. The
    output projection is a :class:`DynamicEmbedding` in decoder mode, which means
    the reconstruction is produced in whatever band set the wavelengths describe.

    Args:
        mask_ratio: Fraction of patches the encoder withheld.
        patch_size: Side of a square patch in pixels.
        encoder_dim: Token width arriving from the encoder.
        dim: Token width inside the decoder.
        depth: Number of transformer layers.
        heads: Attention heads per layer.
        dim_head: Width of one attention head.
        mlp_ratio: Feed-forward expansion factor.
    """

    def __init__(  # noqa: PLR0913
        self,
        mask_ratio: float,
        patch_size: int,
        encoder_dim: int,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        self.dim = dim

        self.enc_to_dec = (
            nn.Linear(encoder_dim, dim) if encoder_dim != dim else nn.Identity()
        )
        self.mask_patch = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=int(dim * mlp_ratio),
        )
        self.embed_to_pixels = DynamicEmbedding(
            wave_dim=WAVE_DIM,
            num_latent_tokens=NUM_LATENT_TOKENS,
            patch_size=patch_size,
            embed_dim=dim,
            is_decoder=True,
        )

    def reconstruct_and_add_encoding(  # noqa: PLR0913
        self,
        unmasked_patches: torch.Tensor,
        unmasked_indices: torch.Tensor,
        masked_indices: torch.Tensor,
        num_patches: int,
        time: torch.Tensor,
        latlon: torch.Tensor,
        gsd: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter kept and mask tokens back into grid order.

        Args:
            unmasked_patches: Encoder output ``(B, 1 + kept, D)``, class token first.
            unmasked_indices: Grid positions of the kept patches.
            masked_indices: Grid positions of the withheld patches.
            num_patches: Total patches in the grid.
            time: Normalised week and hour, ``(B, 4)``.
            latlon: Normalised latitude and longitude, ``(B, 4)``.
            gsd: Scalar ground sample distance in metres.

        Returns:
            Decoder input ``(B, 1 + num_patches, D)``.
        """
        cls_tokens, unmasked_patches = (
            unmasked_patches[:, :1, :],
            unmasked_patches[:, 1:, :],
        )
        device = unmasked_patches.device
        batch = unmasked_patches.shape[0]

        encoding = position_metadata_encoding(
            num_patches=num_patches,
            dim=self.dim,
            gsd=gsd,
            time=time,
            latlon=latlon,
            device=device,
        )
        batch_indices = torch.arange(batch, device=device).unsqueeze(1)

        masked_patches = self.mask_patch.expand(batch, masked_indices.shape[1], -1)
        masked_patches = masked_patches + encoding[batch_indices, masked_indices, :]
        unmasked_patches = (
            unmasked_patches + encoding[batch_indices, unmasked_indices, :]
        )

        decoder_patches = torch.zeros(
            (batch, num_patches, self.dim), device=device, dtype=unmasked_patches.dtype
        )
        decoder_patches[batch_indices, unmasked_indices, :] = unmasked_patches
        decoder_patches[batch_indices, masked_indices, :] = masked_patches
        return torch.cat((cls_tokens, decoder_patches), dim=1)

    def forward(  # noqa: PLR0913
        self,
        encoded_unmasked_patches: torch.Tensor,
        unmasked_indices: torch.Tensor,
        masked_indices: torch.Tensor,
        num_patches: int,
        time: torch.Tensor,
        latlon: torch.Tensor,
        gsd: torch.Tensor,
        waves: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode the encoder output back to pixels.

        Args:
            encoded_unmasked_patches: Encoder output ``(B, 1 + kept, D_enc)``.
            unmasked_indices: Grid positions of the kept patches.
            masked_indices: Grid positions of the withheld patches.
            num_patches: Total patches in the grid.
            time: Normalised week and hour, ``(B, 4)``.
            latlon: Normalised latitude and longitude, ``(B, 4)``.
            gsd: Scalar ground sample distance in metres.
            waves: Band centres ``(C,)`` of the bands to reconstruct.

        Returns:
            Reconstructed patches ``(B, num_patches, C * patch * patch)`` and the
            wavelength embedding used to produce them.
        """
        patches = self.enc_to_dec(encoded_unmasked_patches)
        patches = self.reconstruct_and_add_encoding(
            patches,
            unmasked_indices,
            masked_indices,
            num_patches,
            time,
            latlon,
            gsd,
        )
        decoded = self.transformer(patches)
        pixels, waves = self.embed_to_pixels(decoded, waves)
        return pixels[:, 1:, :], waves
