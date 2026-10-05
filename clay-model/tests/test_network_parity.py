"""Encoder and decoder must reproduce the original end to end."""

from __future__ import annotations

import pytest
import torch

from clay.model.decoder import Decoder
from clay.model.encoder import Encoder

# This file compares the ported `clay` package against the vendored original
# Lightning implementation in clay-outoflightning/clayrepo/clay-foundation-model/
# (import name `claymodel`), which this reproduction package intentionally does
# not include (too large, separate upstream licensing, not needed to reproduce
# the paper's own frozen-encoder-plus-linear-probe numbers). Run these tests
# from the original clay-outoflightning/ checkout, where clayrepo/ is present,
# to use it as a parity oracle.
try:
    import claymodel.model as reference_model
except ImportError:
    pytestmark = pytest.mark.skip(
        reason="claymodel (clay-outoflightning/clayrepo/clay-foundation-model/) "
        "is not part of this reproduction package; see module docstring"
    )

PATCH_SIZE = 8
ENCODER_DIM = 96
DECODER_DIM = 64
DEPTH = 2
HEADS = 4
DIM_HEAD = 24
MLP_RATIO = 2
MASK_RATIO = 0.75
NUM_PATCHES = 16
SEED = 7


def _encoders(mask_ratio: float, shuffle: bool):
    torch.manual_seed(SEED)
    reference = reference_model.Encoder(
        mask_ratio=mask_ratio,
        patch_size=PATCH_SIZE,
        shuffle=shuffle,
        dim=ENCODER_DIM,
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD,
        mlp_ratio=MLP_RATIO,
    ).eval()
    ported = Encoder(
        mask_ratio=mask_ratio,
        patch_size=PATCH_SIZE,
        shuffle=shuffle,
        dim=ENCODER_DIM,
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD,
        mlp_ratio=MLP_RATIO,
    ).eval()
    ported.load_state_dict(reference.state_dict())
    return reference, ported


def _run(model, cube):
    torch.manual_seed(SEED)
    with torch.no_grad():
        return model(cube)


def test_encoder_matches_reference_unmasked(datacube):
    reference, ported = _encoders(mask_ratio=0.0, shuffle=False)
    reference_out = _run(reference, datacube)
    ported_out = _run(ported, datacube)
    for got, want in zip(ported_out, reference_out):
        torch.testing.assert_close(got, want)


def test_encoder_matches_reference_when_masking(datacube):
    reference, ported = _encoders(mask_ratio=MASK_RATIO, shuffle=True)
    reference_out = _run(reference, datacube)
    ported_out = _run(ported, datacube)
    for got, want in zip(ported_out, reference_out):
        torch.testing.assert_close(got, want)


def test_decoder_matches_reference(datacube):
    reference_encoder, ported_encoder = _encoders(mask_ratio=MASK_RATIO, shuffle=True)
    tokens, unmasked, masked, mask_matrix = _run(ported_encoder, datacube)

    torch.manual_seed(SEED)
    reference = reference_model.Decoder(
        mask_ratio=MASK_RATIO,
        patch_size=PATCH_SIZE,
        encoder_dim=ENCODER_DIM,
        dim=DECODER_DIM,
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD,
        mlp_ratio=MLP_RATIO,
    ).eval()
    ported = Decoder(
        mask_ratio=MASK_RATIO,
        patch_size=PATCH_SIZE,
        encoder_dim=ENCODER_DIM,
        dim=DECODER_DIM,
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD,
        mlp_ratio=MLP_RATIO,
    ).eval()
    ported.load_state_dict(reference.state_dict())

    with torch.no_grad():
        reference_pixels, reference_waves = reference(
            tokens,
            unmasked,
            masked,
            mask_matrix,
            datacube["time"],
            datacube["latlon"],
            datacube["gsd"],
            datacube["waves"],
        )
        ported_pixels, ported_waves = ported(
            tokens,
            unmasked,
            masked,
            NUM_PATCHES,
            datacube["time"],
            datacube["latlon"],
            datacube["gsd"],
            datacube["waves"],
        )
    torch.testing.assert_close(ported_pixels, reference_pixels)
    torch.testing.assert_close(ported_waves, reference_waves)
