"""Ported submodules must reproduce the original given the same weights."""

from __future__ import annotations

import pytest
import torch

from clay.constants import NUM_LATENT_TOKENS, WAVE_DIM
from clay.model.dynamic_embedding import DynamicEmbedding
from clay.model.transformer import Transformer

# This file compares the ported `clay` package against the vendored original
# Lightning implementation in clay-outoflightning/clayrepo/clay-foundation-model/
# (import name `claymodel`), which this reproduction package intentionally does
# not include (too large, separate upstream licensing, not needed to reproduce
# the paper's own frozen-encoder-plus-linear-probe numbers). Run these tests
# from the original clay-outoflightning/ checkout, where clayrepo/ is present,
# to use it as a parity oracle.
try:
    import claymodel.backbone as reference_backbone
    import claymodel.factory as reference_factory
except ImportError:
    pytestmark = pytest.mark.skip(
        reason="claymodel (clay-outoflightning/clayrepo/clay-foundation-model/) "
        "is not part of this reproduction package; see module docstring"
    )

PATCH_SIZE = 8
EMBED_DIM = 96
DEPTH = 2
HEADS = 4
DIM_HEAD = 24
MLP_DIM = 192
WAVES = torch.tensor([0.493, 0.56, 0.665])


def test_transformer_matches_reference():
    torch.manual_seed(0)
    reference = reference_backbone.Transformer(
        dim=EMBED_DIM,
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD,
        mlp_dim=MLP_DIM,
        fused_attn=True,
    ).eval()
    ported = Transformer(
        dim=EMBED_DIM, depth=DEPTH, heads=HEADS, dim_head=DIM_HEAD, mlp_dim=MLP_DIM
    ).eval()
    ported.load_state_dict(reference.state_dict())

    x = torch.randn(2, 17, EMBED_DIM)
    with torch.no_grad():
        torch.testing.assert_close(ported(x), reference(x))


def test_patch_embedding_matches_reference():
    torch.manual_seed(0)
    reference = reference_factory.DynamicEmbedding(
        wave_dim=WAVE_DIM,
        num_latent_tokens=NUM_LATENT_TOKENS,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        is_decoder=False,
    ).eval()
    ported = DynamicEmbedding(
        wave_dim=WAVE_DIM,
        num_latent_tokens=NUM_LATENT_TOKENS,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        is_decoder=False,
    ).eval()
    ported.load_state_dict(reference.state_dict())

    pixels = torch.randn(2, WAVES.shape[0], 32, 32)
    with torch.no_grad():
        ported_patches, ported_waves = ported(pixels, WAVES)
        reference_patches, reference_waves = reference(pixels, WAVES)
    torch.testing.assert_close(ported_patches, reference_patches)
    torch.testing.assert_close(ported_waves, reference_waves)


def test_pixel_projection_matches_reference():
    torch.manual_seed(0)
    reference = reference_factory.DynamicEmbedding(
        wave_dim=WAVE_DIM,
        num_latent_tokens=NUM_LATENT_TOKENS,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        is_decoder=True,
    ).eval()
    ported = DynamicEmbedding(
        wave_dim=WAVE_DIM,
        num_latent_tokens=NUM_LATENT_TOKENS,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        is_decoder=True,
    ).eval()
    ported.load_state_dict(reference.state_dict())

    tokens = torch.randn(2, 16, EMBED_DIM)
    with torch.no_grad():
        ported_pixels, _ = ported(tokens, WAVES)
        reference_pixels, _ = reference(tokens, WAVES)
    torch.testing.assert_close(ported_pixels, reference_pixels)
