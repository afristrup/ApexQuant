"""Level sets are antisymmetric, and quantization commutes with negation.

Every density these codebooks are fitted to is symmetric, so the optimal level
set satisfies ``levels[i] == -levels[n-1-i]`` and the quantizer commutes with
negation. Before `symmetrize_levels` the grid solvers missed that by about one
grid step, $5 \\times 10^{-5}$, because :func:`torch.bucketize` sends a grid point
lying exactly on a cell boundary to the cell above it and the mirror point to the
cell above *its* boundary.

The size of the residue is not the point; it is invisible in the distortion,
moving ``rho**2`` by $2 \\times 10^{-8}$. The point is that the residual recursion
amplifies it. A coordinate near a cell boundary flips to the other codeword, and
the next pass works on a residual an order of magnitude smaller, so an absolute
$10^{-5}$ became a few percent of relative difference by the second pass.

The consequence worth a regression test is folding a batch normalization into the
preceding convolution. With a positive fold constant it commutes with
quantization exactly, because normalizing a group onto the unit sphere divides any
positive per-row scalar out. With a negative one the row's sign flips, and an
asymmetric level set quantizes it differently: 103 of torchvision ResNet-50's
26,560 batch-norm channels have negative gamma, and 392 of EfficientNet-B0's
21,008.

These assertions are exact where exactness is available. Antisymmetry holds bit
for bit because ``out[n-1-i]`` evaluates the same subtraction negated, and the
round trip then commutes with negation bit for bit as well; only the fold test
carries a tolerance, because scaling a row really does change the arithmetic.
"""

import pytest
import torch

from apexquant.codebook import (apex_beta_levels, generalized_gaussian_levels,
                                       lloyd_max_levels, lp_levels, symmetrize_levels,
                                       uniform_levels)
from apexquant.config import ApexQuantConfig
from apexquant.quantize.dequantize_weight import dequantize_weight
from apexquant.quantize.turboquantize_weight import turboquantize_weight

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

BITS = (2, 4, 8)
FAN_INS = (64, 128, 256, 512)
PASSES = (1, 2, 3)
CODEBOOKS = ("apex", "gaussian", "uniform", "lp")
# The spec's bound on the round trip. The assertions below beat it outright, so a
# failure means a real regression rather than drift against a tight threshold.
EQUIVARIANCE_TOL = 1e-6
# Scaling a row by a constant is not a bit-exact operation, so the fold test
# compares at float32 round-off rather than exactly.
FOLD_TOL = 1e-5
ROWS, COLS, GROUP = 8, 512, 256
NEGATIVE_ROWS = 3          # rows given a negative fold constant, as a negative gamma does


def _solvers(bits, fan_in):
    return {
        "apex_beta": apex_beta_levels(bits, fan_in),
        "lloyd_max": lloyd_max_levels(bits),
        "lp": lp_levels(bits, fan_in, 0.7),
        "generalized_gaussian": generalized_gaussian_levels(bits, 2.0),
        "uniform": uniform_levels(bits),
    }


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("fan_in", FAN_INS)
def test_levels_are_exactly_antisymmetric(bits, fan_in):
    for name, levels in _solvers(bits, fan_in).items():
        residue = (levels + levels.flip(0)).abs().max().item()
        assert residue == 0.0, f"{name}(bits={bits}, fan_in={fan_in}) off by {residue:.3e}"


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("fan_in", FAN_INS)
def test_levels_stay_strictly_ascending(bits, fan_in):
    """Symmetrizing must not disturb the ordering every caller relies on.

    The midpoint boundaries in `quantize_to_indices` and `torch.bucketize` are
    only cell boundaries if the levels ascend.
    """
    for name, levels in _solvers(bits, fan_in).items():
        assert torch.all(levels[1:] > levels[:-1]), f"{name} not strictly ascending"


def test_symmetrize_is_idempotent_and_fixes_an_asymmetric_input():
    v = torch.tensor([-0.9, -0.3, 0.2, 0.8])
    once = symmetrize_levels(v)
    assert torch.equal(once, symmetrize_levels(once))
    assert (once + once.flip(0)).abs().max().item() == 0.0
    # An odd-length set has one element that is its own mirror; it must be zero.
    odd = symmetrize_levels(torch.tensor([-1.0, 0.1, 1.0]))
    assert odd[1].item() == 0.0


@needs_cuda
@pytest.mark.parametrize("codebook", CODEBOOKS)
@pytest.mark.parametrize("passes", PASSES)
def test_round_trip_commutes_with_negation(codebook, passes):
    """``-dequantize(quantize(W))`` against ``dequantize(quantize(-W))``."""
    w = torch.randn(ROWS, COLS, generator=torch.Generator(device="cuda").manual_seed(0),
                    device="cuda")
    cfg = ApexQuantConfig(bits=2, passes=passes, group_size=GROUP, codebook=codebook)
    negated_first = dequantize_weight(turboquantize_weight(-w, cfg))
    quantized_first = -dequantize_weight(turboquantize_weight(w, cfg))
    rel = ((quantized_first - negated_first).norm() / negated_first.norm()).item()
    assert rel < EQUIVARIANCE_TOL, f"{codebook} at P={passes}: rel diff {rel:.3e}"


@needs_cuda
@pytest.mark.parametrize("passes", PASSES)
def test_negative_fold_constants_commute_with_quantization(passes):
    """A batch-norm fold commutes whatever the sign of gamma.

    This is the case the asymmetry actually broke. A positive fold constant
    divides out of the group's L2 normalizer; a negative one flips the row, which
    only lands on the same codewords if the level set is antisymmetric.
    """
    gen = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn(ROWS, COLS, generator=gen, device="cuda")
    fold = torch.rand(ROWS, generator=gen, device="cuda") + 0.25
    fold[:NEGATIVE_ROWS] *= -1
    cfg = ApexQuantConfig(bits=2, passes=passes, group_size=GROUP, codebook="apex")

    def recon(x):
        return dequantize_weight(turboquantize_weight(x, cfg))

    fold_then_quantize = recon(fold[:, None] * w)
    quantize_then_fold = fold[:, None] * recon(w)
    rel = ((fold_then_quantize - quantize_then_fold).norm()
           / quantize_then_fold.norm()).item()
    assert rel < FOLD_TOL, f"fold does not commute at P={passes}: rel diff {rel:.3e}"
