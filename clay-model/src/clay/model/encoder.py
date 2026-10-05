"""The Clay encoder: dynamic patch embedding, masking, transformer trunk."""

from __future__ import annotations

import torch
from torch import nn

from ..constants import NUM_LATENT_TOKENS, TOKEN_INIT_STD, WAVE_DIM
from .dynamic_embedding import DynamicEmbedding
from .position_metadata_encoding import position_metadata_encoding
from .transformer import Transformer


class Encoder(nn.Module):
    """Encode a datacube into patch tokens, dropping a fraction of them.

    At inference ``mask_ratio`` is 0 and ``shuffle`` is off, so every patch
    survives in grid order and the returned index tensors are trivial; the
    masking machinery is kept because it is what the published checkpoints were
    trained with and reconstruction still needs it.

    Args:
        mask_ratio: Fraction of patches withheld from the transformer.
        patch_size: Side of a square patch in pixels.
        shuffle: Draw the masked subset at random rather than taking a prefix.
        dim: Token width.
        depth: Number of transformer layers.
        heads: Attention heads per layer.
        dim_head: Width of one attention head.
        mlp_ratio: Feed-forward expansion factor.
    """

    def __init__(  # noqa: PLR0913
        self,
        mask_ratio: float,
        patch_size: int,
        shuffle: bool,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.dim = dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * TOKEN_INIT_STD)

        self.patch_embedding = DynamicEmbedding(
            wave_dim=WAVE_DIM,
            num_latent_tokens=NUM_LATENT_TOKENS,
            patch_size=patch_size,
            embed_dim=dim,
            is_decoder=False,
        )
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=int(dim * mlp_ratio),
        )

    def mask_out(
        self, patches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the patches into a kept subset and the indices of the rest.

        Args:
            patches: Patch tokens ``(B, L, D)``.

        Returns:
            The kept patches ``(B, L * (1 - mask_ratio), D)``, the kept indices,
            the masked indices, and a ``(B, L)`` matrix that is 1 where a patch
            was masked out.
        """
        batch, num_patches, _ = patches.shape
        if self.shuffle:
            noise = torch.randn((batch, num_patches), device=patches.device)
        else:
            noise = torch.arange(batch * num_patches, device=patches.device).view(
                batch, num_patches
            )

        random_indices = torch.argsort(noise, dim=-1)
        reverse_indices = torch.argsort(random_indices, dim=-1)

        num_masked = int(self.mask_ratio * num_patches)
        masked_indices = random_indices[:, :num_masked]
        unmasked_indices = random_indices[:, num_masked:]

        masked_matrix = torch.zeros((batch, num_patches), device=patches.device)
        masked_matrix[:, :num_masked] = 1
        masked_matrix = torch.gather(masked_matrix, dim=1, index=reverse_indices)

        batch_indices = torch.arange(batch, device=patches.device).unsqueeze(1)
        unmasked_patches = patches[batch_indices, unmasked_indices, :]
        return unmasked_patches, unmasked_indices, masked_indices, masked_matrix

    def forward(
        self, datacube: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode one datacube.

        Args:
            datacube: ``pixels`` ``(B, C, H, W)``, ``time`` ``(B, 4)``, ``latlon``
                ``(B, 4)``, scalar ``gsd``, and ``waves`` ``(C,)`` band centres.

        Returns:
            Encoded tokens ``(B, 1 + kept, D)`` with the class token first, plus
            the kept indices, the masked indices and the mask matrix.
        """
        cube = datacube["pixels"]
        batch = cube.shape[0]

        patches, _ = self.patch_embedding(cube, datacube["waves"])
        patches = patches + position_metadata_encoding(
            num_patches=patches.shape[1],
            dim=self.dim,
            gsd=datacube["gsd"],
            time=datacube["time"],
            latlon=datacube["latlon"],
            device=patches.device,
        )

        unmasked, unmasked_indices, masked_indices, masked_matrix = self.mask_out(
            patches
        )
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        unmasked = torch.cat((cls_tokens, unmasked), dim=1)

        encoded = self.transformer(unmasked)
        return encoded, unmasked_indices, masked_indices, masked_matrix
