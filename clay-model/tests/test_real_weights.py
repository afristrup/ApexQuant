"""The published Clay v1.5 weights, through the TF32 kernels.

Skipped unless the checkpoint is already in the local Hugging Face cache; these
tests never reach the network. Fetch it once with::

    huggingface-cli download made-with-clay/Clay v1.5/clay-v1.5.ckpt
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from huggingface_hub import try_to_load_from_cache

from clay.constants import DISABLE_TRITON_ENV
from clay.inference.embed import embed
from clay.inference.load_checkpoint_state_dict import load_checkpoint_state_dict
from clay.inference.load_sensor_metadata import load_sensor_metadata
from clay.model.build_encoder import build_encoder

REPO = "made-with-clay/Clay"
CHECKPOINT = "v1.5/clay-v1.5.ckpt"
# Apache-2.0 sensor metadata (band wavelengths/GSD per platform), copied in as a
# standalone data fixture; it is not part of the excluded vendored Lightning
# implementation (clay-outoflightning/clayrepo/clay-foundation-model/), just data
# that lived alongside it and that `load_sensor_metadata` needs to be exercised.
METADATA = Path(__file__).resolve().parent.parent / "configs" / "metadata.yaml"
PLATFORM = "sentinel-2-l2a"
CHIP = 256
BATCH = 2
# Chip embeddings are consumed by similarity search, so direction is what has to
# survive the format change, not the last mantissa bits.
MIN_COSINE_SIMILARITY = 0.9999

cached = try_to_load_from_cache(REPO, CHECKPOINT)
pytestmark = [
    pytest.mark.skipif(
        not isinstance(cached, str), reason=f"{REPO}/{CHECKPOINT} is not cached"
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="the Triton kernels need a CUDA device"
    ),
]


@pytest.fixture(scope="module")
def clay_encoder():
    """The published encoder, on the GPU, in eval mode."""
    hyper_parameters = torch.load(cached, map_location="cpu", weights_only=False).get(
        "hyper_parameters", {}
    )
    encoder = build_encoder(
        hyper_parameters.get("model_size", "large"),
        patch_size=hyper_parameters.get("patch_size", 8),
    ).eval()
    encoder.load_state_dict(load_checkpoint_state_dict(cached))
    return encoder.cuda()


@pytest.fixture(scope="module")
def sentinel_cube():
    """A batch of smooth fields: real normalised chips are spatially correlated."""
    waves, gsd = load_sensor_metadata(METADATA, PLATFORM)
    torch.manual_seed(0)
    coarse = torch.randn(BATCH, waves.shape[0], CHIP // 16, CHIP // 16)
    return {
        "pixels": F.interpolate(coarse, size=(CHIP, CHIP), mode="bicubic").cuda(),
        "time": torch.randn(BATCH, 4, device="cuda"),
        "latlon": torch.randn(BATCH, 4, device="cuda"),
        "gsd": gsd,
        "waves": waves,
    }


def test_published_weights_load_into_the_port(clay_encoder):
    """Every published encoder tensor must land on a parameter of the port."""
    published = load_checkpoint_state_dict(cached)
    assert set(published) == set(clay_encoder.state_dict())


def test_tf32_kernels_preserve_the_chip_embedding(
    clay_encoder, sentinel_cube, monkeypatch
):
    """The TF32 path must return the same embedding direction as torch."""
    monkeypatch.setenv(DISABLE_TRITON_ENV, "1")
    reference = embed(clay_encoder, sentinel_cube)[:, 0, :]

    monkeypatch.delenv(DISABLE_TRITON_ENV)
    triton = embed(clay_encoder, sentinel_cube)[:, 0, :]

    similarity = F.cosine_similarity(triton, reference).min().item()
    assert similarity > MIN_COSINE_SIMILARITY, similarity
