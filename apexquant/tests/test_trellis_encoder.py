"""The fused trellis encoder against the torch Viterbi it replaces.

Codes are deliberately not asserted equal. Ties in the ACS minimum break to
whichever index a reduction happens to keep and the two implementations do not
agree on that, so the assertion that means something is on the *metric*: a tie
means two winning paths, and a winning path is a winning path. Per-row squared
error is what these tests compare, which fails on any real search error and is
indifferent to which of several optima was returned.
"""

import pytest
import torch

from apexquant.vq import trellis_codec as tc
from apexquant.vq.trellis_codec import TrellisCodec

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="CUDA required")
# APEXQUANT_NO_TRITON=1 turns the kernel off deliberately; the tests that exist
# to compare it against the reference have nothing to compare then.
needs_fused = pytest.mark.skipif(
    not (torch.cuda.is_available() and tc.HAVE_TRITON),
    reason="fused ACS kernel required")


def _sphere(n, d, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    v = torch.randn(n, d, generator=g, device=device)
    return v / v.norm(dim=1, keepdim=True)


def _row_err(codec, u, codes):
    """Squared error per row for a symbol sequence, in the codec's own units."""
    return (u - codec._walk(codes.to(torch.int64)) * codec.scale).pow(2).sum(1)


@needs_fused
@pytest.mark.parametrize("trellis_bits", [10, 12, 14])
def test_fused_acs_matches_reference_metric(trellis_bits):
    codec = TrellisCodec(d=256, bits=2, trellis_bits=trellis_bits,
                         device="cuda")
    u = _sphere(512, 256, "cuda")
    x = u / codec.scale
    fused = _row_err(codec, u, codec._acs_codes(x))
    ref = _row_err(codec, u, codec._viterbi_codes(x))
    assert torch.allclose(fused, ref, atol=1e-5, rtol=1e-4)


@needs_fused
def test_encode_dispatches_to_the_fused_path():
    """Guard the guard: a silent fallback would make the test above vacuous."""
    codec = TrellisCodec(d=256, bits=2, trellis_bits=12, device="cuda")
    assert tc.HAVE_TRITON and codec.S <= tc._ACS_MAX_STATES
    # Above the register budget the torch path must still be reachable.
    wide = TrellisCodec(d=256, bits=2, trellis_bits=16, device="cuda")
    assert wide.S > tc._ACS_MAX_STATES


@needs_cuda
def test_wide_trellis_falls_back_and_still_encodes():
    codec = TrellisCodec(d=64, bits=2, trellis_bits=16, device="cuda")
    u = _sphere(8, 64, "cuda")
    assert (u - codec(u)).pow(2).sum().item() / 8 < 0.15


@needs_cuda
def test_encode_round_trip_is_a_reconstruction():
    codec = TrellisCodec(d=512, bits=2, trellis_bits=12, device="cuda")
    u = _sphere(128, 512, "cuda", seed=1)
    rel = (u - codec.decode(codec.encode(u))).norm() / u.norm()
    # rho^2 ~ 0.075 at this rate, so the relative error sits near sqrt of it.
    assert 0.2 < rel.item() < 0.35


@needs_cuda
def test_beam_still_takes_the_beam_path():
    """`beam` is the instrument the encoder-lever sweep is made with."""
    codec = TrellisCodec(d=256, bits=2, trellis_bits=12, device="cuda", beam=32)
    u = _sphere(64, 256, "cuda")
    exact = TrellisCodec(d=256, bits=2, trellis_bits=12, device="cuda")
    assert (u - codec(u)).pow(2).sum() >= (u - exact(u)).pow(2).sum()


# --------------------------------------------------------------------------
# 1MAD-p: the codeword marginal is a lever, the trellis is not.
# --------------------------------------------------------------------------

def test_shape_two_is_bit_identical_to_the_published_table():
    # Every artifact published so far was encoded at the Gaussian target. If the
    # default ever stops reproducing it exactly, those artifacts decode wrong.
    import torch

    from apexquant.vq.trellis_codec import _codeword_table
    default = _codeword_table(12, 0, torch.device("cpu"))
    explicit = _codeword_table(12, 0, torch.device("cpu"), 2.0)
    assert torch.equal(default, explicit)


def test_shape_controls_the_codeword_tail():
    import torch

    from apexquant.vq.trellis_codec import _codeword_table

    def kurtosis(cw):
        return (cw.pow(4).mean() / cw.pow(2).mean() ** 2).item()

    tables = {s: _codeword_table(12, 0, torch.device("cpu"), s) for s in (2.0, 1.0, 0.7)}
    for cw in tables.values():
        assert abs(cw.std().item() - 1.0) < 1e-3      # unit variance throughout
    # Gaussian is 3, Laplace is 6, and below 1 is heavier than either.
    assert abs(kurtosis(tables[2.0]) - 3.0) < 0.3
    assert abs(kurtosis(tables[1.0]) - 6.0) < 0.6
    assert kurtosis(tables[0.7]) > kurtosis(tables[1.0]) > kurtosis(tables[2.0])


def test_shape_is_deterministic_and_device_independent():
    # The artifact records the shape, not the table, so a decoder rebuilding it
    # must land on the same numbers every time.
    import torch

    from apexquant.vq.trellis_codec import _codeword_table
    a = _codeword_table(10, 3, torch.device("cpu"), 0.7)
    b = _codeword_table(10, 3, torch.device("cpu"), 0.7)
    assert torch.equal(a, b)


def test_codec_round_trips_under_a_shaped_marginal():
    import torch

    from apexquant.vq.trellis_codec import TrellisCodec

    torch.manual_seed(0)
    u = torch.randn(4, 256)
    u = u / u.norm(dim=1, keepdim=True)
    for shape in (2.0, 1.0):
        codec = TrellisCodec(d=256, bits=2, trellis_bits=8, shape=shape)
        assert codec.shape == shape
        recon = codec(u)
        assert recon.shape == u.shape
        assert torch.isfinite(recon).all()
        # A working stage removes most of the energy at 2 bits per coordinate.
        assert (u - recon).norm() < 0.75 * u.norm()


def test_shaped_marginal_trades_squared_error_for_a_shorter_tail():
    import torch

    from apexquant.vq.trellis_codec import TrellisCodec

    torch.manual_seed(0)
    u = torch.randn(8, 512)
    u = u / u.norm(dim=1, keepdim=True)
    err = {}
    for shape in (2.0, 1.0):
        codec = TrellisCodec(d=512, bits=2, trellis_bits=8, shape=shape)
        err[shape] = (u - codec(u)).flatten()
    # Heavier-tailed codewords cost mean squared error, by construction.
    assert err[1.0].pow(2).mean() > err[2.0].pow(2).mean()
