"""The Triton path must reproduce the original encoder end to end on CUDA."""

from __future__ import annotations

import pytest
import torch

from clay.model.build_encoder import build_encoder
from clay.model.sizes import SIZES
from clay.ops.use_triton import use_triton

# Two of the four tests below (test_encoder_on_cuda_matches_reference,
# test_class_token_survives_tf32) compare the ported `clay` package against the
# vendored original Lightning implementation in
# clay-outoflightning/clayrepo/clay-foundation-model/ (import name `claymodel`),
# which this reproduction package intentionally does not include (too large,
# separate upstream licensing, not needed to reproduce the paper's own
# frozen-encoder-plus-linear-probe numbers). The whole module is skipped rather
# than just those two tests, since the `import claymodel.model` below would
# otherwise fail at collection time for the other two as well. Run these tests
# from the original clay-outoflightning/ checkout, where clayrepo/ is present,
# to use it as a parity oracle.
try:
    import claymodel.model as reference_model
except ImportError:
    pytestmark = pytest.mark.skip(
        reason="claymodel (clay-outoflightning/clayrepo/clay-foundation-model/) "
        "is not part of this reproduction package; see module docstring"
    )
else:
    pytestmark = pytest.mark.skipif(
        not torch.cuda.is_available(), reason="the Triton kernels need a CUDA device"
    )

SIZE = "tiny"
PATCH_SIZE = 8
CHIP = 64
BANDS = 3
BATCH = 2
SEED = 11
# The shipped kernels accumulate in TF32, which keeps 10 mantissa bits against
# fp32's 23, and the drift compounds over six transformer layers. This is the
# budget for the whole stack; the per-kernel ieee tests are the tight ones.
TOLERANCE = {"rtol": 1e-2, "atol": 5e-3}

# What actually has to survive: two embeddings that point the same way. Clay
# embeddings are consumed by similarity search, not read off elementwise.
MIN_COSINE_SIMILARITY = 0.9999


def _cuda_datacube():
    torch.manual_seed(SEED)
    return {
        "pixels": torch.randn(BATCH, BANDS, CHIP, CHIP, device="cuda"),
        "time": torch.randn(BATCH, 4, device="cuda"),
        "latlon": torch.randn(BATCH, 4, device="cuda"),
        "gsd": torch.tensor(10.0),
        "waves": torch.tensor([0.493, 0.56, 0.665]),
    }


def _reference_encoder(mask_ratio: float, shuffle: bool):
    torch.manual_seed(SEED)
    return reference_model.Encoder(
        mask_ratio=mask_ratio,
        patch_size=PATCH_SIZE,
        shuffle=shuffle,
        **SIZES[SIZE]["encoder"],
    ).eval()


@pytest.mark.parametrize(
    ("mask_ratio", "shuffle"), [(0.0, False), (0.75, True)], ids=["whole", "masked"]
)
def test_encoder_on_cuda_matches_reference(mask_ratio, shuffle):
    reference = _reference_encoder(mask_ratio, shuffle).cuda()
    ported = build_encoder(SIZE, mask_ratio=mask_ratio, shuffle=shuffle).eval().cuda()
    ported.load_state_dict(reference.state_dict())

    cube = _cuda_datacube()
    with torch.no_grad():
        assert use_triton(cube["pixels"]), "expected the Triton path to be taken"
        torch.manual_seed(SEED)
        reference_out = reference(cube)
        torch.manual_seed(SEED)
        ported_out = ported(cube)

    for got, want in zip(ported_out, reference_out):
        torch.testing.assert_close(got, want, **TOLERANCE)


def test_training_mode_falls_back_to_torch():
    """Gradients must never silently vanish into a forward-only kernel."""
    ported = build_encoder(SIZE).cuda()
    cube = _cuda_datacube()
    activation = ported.cls_token * torch.ones(1, device="cuda")
    assert not use_triton(activation)
    assert use_triton(cube["pixels"])


def test_class_token_survives_tf32():
    """The chip embedding must stay the same direction under the TF32 kernels."""
    reference = _reference_encoder(0.0, False).cuda()
    ported = build_encoder(SIZE).eval().cuda()
    ported.load_state_dict(reference.state_dict())

    cube = _cuda_datacube()
    with torch.no_grad():
        want = reference(cube)[0][:, 0, :]
        got = ported(cube)[0][:, 0, :]

    similarity = torch.nn.functional.cosine_similarity(got, want).min().item()
    assert similarity > MIN_COSINE_SIMILARITY, similarity


def test_triton_is_the_default_on_cuda():
    """A CUDA tensor outside autograd must reach the kernels without a flag."""
    assert use_triton(torch.zeros(1, device="cuda"))
