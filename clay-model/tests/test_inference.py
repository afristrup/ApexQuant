"""The inference entry points: shapes, guards, and checkpoint extraction."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from clay.inference.embed import embed
from clay.inference.load_checkpoint_state_dict import load_checkpoint_state_dict
from clay.inference.load_sensor_metadata import load_sensor_metadata
from clay.model.build_encoder import build_encoder
from clay.model.sizes import SIZES

SIZE = "tiny"
PATCH_SIZE = 8
CHIP = 32
# Apache-2.0 sensor metadata (band wavelengths/GSD per platform), copied in as a
# standalone data fixture; it is not part of the excluded vendored Lightning
# implementation (clay-outoflightning/clayrepo/clay-foundation-model/), just data
# that lived alongside it and that `load_sensor_metadata` needs to be exercised.
METADATA = Path(__file__).resolve().parent.parent / "configs" / "metadata.yaml"


def test_embed_returns_a_token_per_patch_plus_the_class_token(datacube):
    encoder = build_encoder(SIZE).eval()
    tokens = embed(encoder, datacube)
    patches = (CHIP // PATCH_SIZE) ** 2
    assert tokens.shape == (2, 1 + patches, SIZES[SIZE]["encoder"]["dim"])


def test_embed_refuses_a_masking_encoder(datacube):
    encoder = build_encoder(SIZE, mask_ratio=0.75).eval()
    with pytest.raises(ValueError, match="mask_ratio"):
        embed(encoder, datacube)


def test_checkpoint_extraction_drops_the_training_only_weights(tmp_path):
    encoder = build_encoder(SIZE)
    checkpoint = {
        "state_dict": {
            **{f"model.encoder.{k}": v for k, v in encoder.state_dict().items()},
            "model.teacher.blocks.0.norm1.weight": torch.zeros(4),
            "model.proj.weight": torch.zeros(4, 4),
        }
    }
    path = tmp_path / "clay.ckpt"
    torch.save(checkpoint, path)

    extracted = load_checkpoint_state_dict(path)
    assert set(extracted) == set(encoder.state_dict())
    build_encoder(SIZE).load_state_dict(extracted)


def test_checkpoint_extraction_fails_loudly_on_an_unknown_prefix(tmp_path):
    path = tmp_path / "clay.ckpt"
    torch.save({"state_dict": {"model.encoder.cls_token": torch.zeros(1)}}, path)
    with pytest.raises(ValueError, match="no weights under prefix"):
        load_checkpoint_state_dict(path, prefix="model.decoder.")


def test_sensor_metadata_matches_the_band_count():
    waves, gsd = load_sensor_metadata(METADATA, "sentinel-2-l2a")
    assert waves.shape == (10,)
    assert gsd.item() == 10
