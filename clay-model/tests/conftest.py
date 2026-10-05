"""Make the ported `clay` package importable, and pin float32 precision."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# The port deliberately sets no global precision policy, unlike the vendored
# original (clay-outoflightning/clayrepo/clay-foundation-model/'s claymodel.model
# sets this to "medium" at import time, silently turning on TF32 for every
# matmul). Pin the highest precision here so CPU-side tests are directly
# comparable to plain fp32 arithmetic.
torch.set_float32_matmul_precision("highest")


@pytest.fixture
def datacube():
    """A small random datacube in the dict layout the encoder expects."""
    torch.manual_seed(0)
    bands = 3
    return {
        "pixels": torch.randn(2, bands, 32, 32),
        "time": torch.randn(2, 4),
        "latlon": torch.randn(2, 4),
        "gsd": torch.tensor(10.0),
        "waves": torch.tensor([0.493, 0.56, 0.665]),
    }
