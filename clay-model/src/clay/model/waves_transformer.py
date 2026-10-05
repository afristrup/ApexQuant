"""Generate patch-projection weights from band wavelengths (DOFA-style)."""

from __future__ import annotations

import torch
from torch import nn

from ..constants import (
    TOKEN_INIT_STD,
    WAVES_TRANSFORMER_HEADS,
    WAVES_TRANSFORMER_LAYERS,
)


class WavesTransformer(nn.Module):
    """Turn a set of band wavelengths into the weights of a patch projection.

    The learned latent tokens are prepended to the wavelength tokens and a bias
    token is appended; after one transformer layer, the wavelength positions are
    read back out as per-band projection weights and the bias position as the
    projection bias. Because the weights are produced rather than stored, the
    same parameters serve any sensor, including ones unseen in training.
    """

    def __init__(  # noqa: PLR0913
        self,
        wave_dim: int,
        output_dim: int,
        num_latent_tokens: int,
        embed_dim: int,
        is_decoder: bool,
        num_heads: int = WAVES_TRANSFORMER_HEADS,
        num_layers: int = WAVES_TRANSFORMER_LAYERS,
    ) -> None:
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.is_decoder = is_decoder
        layer = nn.TransformerEncoderLayer(
            d_model=wave_dim,
            nhead=num_heads,
            activation="gelu",
            dropout=0,
            norm_first=False,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

        self.fc_weight = nn.Linear(wave_dim, output_dim)
        self.fc_bias = None if is_decoder else nn.Linear(wave_dim, embed_dim)

        self.weight_tokens = nn.Parameter(
            torch.randn(num_latent_tokens, wave_dim) * TOKEN_INIT_STD
        )
        self.bias_token = nn.Parameter(torch.randn(1, wave_dim) * TOKEN_INIT_STD)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = torch.cat([self.weight_tokens, x, self.bias_token], dim=0)
        out = self.encoder(x)
        band_slice = slice(self.num_latent_tokens, -1)
        weights = self.fc_weight(out[band_slice] + x[band_slice])
        bias = None if self.is_decoder else self.fc_bias(out[-1])
        return weights, bias
