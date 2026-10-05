"""Wavelength-conditioned patch embedding, and its transpose for the decoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..constants import DYNAMIC_WEIGHT_SCALE
from ..ops.sincos_1d import sincos_1d
from .fc_block import FCBlock
from .waves_transformer import WavesTransformer


class DynamicEmbedding(nn.Module):
    """Project pixels to patch tokens with weights generated from wavelengths.

    In encoder mode the generated tensor is reshaped into a convolution kernel of
    stride ``patch_size``, so patching and embedding are a single strided
    convolution. In decoder mode the same generator produces the transposed map
    from token back to ``(bands, patch, patch)`` pixels, and there is no bias.

    Args:
        wave_dim: Width of the wavelength embedding.
        num_latent_tokens: Learned tokens prepended inside the weight generator.
        patch_size: Side of a square patch in pixels.
        embed_dim: Width of a patch token.
        is_decoder: Generate the pixel-producing map instead of the patch one.
    """

    def __init__(  # noqa: PLR0913
        self,
        wave_dim: int,
        num_latent_tokens: int,
        patch_size: int,
        embed_dim: int,
        is_decoder: bool = False,
    ) -> None:
        super().__init__()
        self.wave_dim = wave_dim
        self.num_latent_tokens = num_latent_tokens
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.is_decoder = is_decoder
        self.output_dim = (patch_size**2) * embed_dim

        self.weight_generator = WavesTransformer(
            wave_dim,
            self.output_dim,
            self.num_latent_tokens,
            self.embed_dim,
            is_decoder,
        )
        self.fclayer = FCBlock(self.wave_dim)

        self.initialize_weights()

    def forward(
        self, batch: torch.Tensor, waves: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        waves = sincos_1d(waves.to(batch.device), self.wave_dim)
        waves = self.fclayer(waves)
        weight, bias = self.weight_generator(waves)
        bands = weight.shape[0]
        patch = self.patch_size

        if self.is_decoder:
            # (bands, patch * patch * embed) -> (bands * patch * patch, embed)
            dynamic_weight = weight.view(bands, patch, patch, self.embed_dim).reshape(
                bands * patch * patch, self.embed_dim
            )
            return F.linear(batch, dynamic_weight * DYNAMIC_WEIGHT_SCALE), waves

        # (bands, embed * patch * patch) -> conv kernel (embed, bands, patch, patch)
        dynamic_weight = weight.view(bands, self.embed_dim, patch, patch).permute(
            1, 0, 2, 3
        )
        patches = F.conv2d(
            batch, dynamic_weight * DYNAMIC_WEIGHT_SCALE, bias=bias, stride=patch
        )
        return patches.flatten(2).transpose(1, 2), waves

    def initialize_weights(self) -> None:
        """Xavier-initialise every linear and convolution, zeroing the biases."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
