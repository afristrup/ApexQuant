"""The position encodings must match the original bit for bit."""

from __future__ import annotations

import pytest
import torch

from clay.constants import METADATA_DIM
from clay.ops.sincos_1d import sincos_1d
from clay.ops.sincos_2d_with_gsd import sincos_2d_with_gsd

# This file compares the ported `clay` package against the vendored original
# Lightning implementation in clay-outoflightning/clayrepo/clay-foundation-model/
# (import name `claymodel`), which this reproduction package intentionally does
# not include (too large, separate upstream licensing, not needed to reproduce
# the paper's own frozen-encoder-plus-linear-probe numbers). Run these tests
# from the original clay-outoflightning/ checkout, where clayrepo/ is present,
# to use it as a parity oracle.
try:
    from claymodel.utils import posemb_sincos_1d, posemb_sincos_2d_with_gsd
except ImportError:
    pytestmark = pytest.mark.skip(
        reason="claymodel (clay-outoflightning/clayrepo/clay-foundation-model/) "
        "is not part of this reproduction package; see module docstring"
    )

WAVE_DIM = 128
ENCODER_DIM = 768


def test_sincos_1d_matches_reference():
    waves = torch.tensor([0.493, 0.56, 0.665, 0.704, 2.19])
    assert torch.equal(sincos_1d(waves, WAVE_DIM), posemb_sincos_1d(waves, WAVE_DIM))


def test_sincos_2d_with_gsd_matches_reference():
    grid = 8
    dim = ENCODER_DIM - METADATA_DIM
    gsd = torch.tensor(10.0)
    ported = sincos_2d_with_gsd(grid, grid, dim, gsd, torch.device("cpu"))
    reference = posemb_sincos_2d_with_gsd(h=grid, w=grid, dim=dim, gsd=gsd)
    assert torch.equal(ported, reference)


def test_sincos_2d_rejects_width_that_is_not_a_multiple_of_four():
    try:
        sincos_2d_with_gsd(4, 4, 10, torch.tensor(1.0), torch.device("cpu"))
    except ValueError:
        return
    raise AssertionError("expected a ValueError for a width that is not 4-divisible")
