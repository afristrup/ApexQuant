"""Nested decoding: a stored artifact's prefix is a fresh shallow quantize.

``serving_evals/prefix_check.py`` checks this bit-identity on real HF
checkpoints; this is the same claim against synthetic weights, needing no
download. Proposition 2's recursion is purely sequential, so the codes and
scale computed for pass ``p`` depend only on the residual left by passes
``0..p-1``, not on how many passes the caller ultimately asks for -- reading
the first ``P'`` passes of a ``P``-pass artifact must therefore match a fresh
``P'``-pass quantize exactly, not just approximately.
"""

import torch

from apexquant.vq.progressive_artifact import dequantize, quantize
from apexquant.vq.scalar_codec import ScalarCodec

BITS = 2
GROUP_SIZE = 64
OUT_FEATURES = 12
IN_FEATURES = 3 * GROUP_SIZE
SEED = 0
FULL_PASSES = 3
PREFIX_PASSES = (1, 2)


def _codec():
    return ScalarCodec(d=GROUP_SIZE, bits=BITS)


def _weight():
    torch.manual_seed(0)
    return torch.randn(OUT_FEATURES, IN_FEATURES)


def test_prefix_codes_and_scale_are_bit_identical_to_a_fresh_shallow_quantize():
    w = _weight()
    codes_full, scale_full = quantize(w, _codec(), FULL_PASSES, group_size=GROUP_SIZE, seed=SEED)

    for prefix in PREFIX_PASSES:
        codes_fresh, scale_fresh = quantize(w, _codec(), prefix, group_size=GROUP_SIZE, seed=SEED)
        assert torch.equal(codes_full[:prefix], codes_fresh)
        assert torch.equal(scale_full[:prefix], scale_fresh)


def test_prefix_dequantize_matches_a_fresh_shallow_dequantize():
    w = _weight()
    codes, scale = quantize(w, _codec(), FULL_PASSES, group_size=GROUP_SIZE, seed=SEED)

    for prefix in PREFIX_PASSES:
        from_prefix = dequantize(codes, scale, _codec(), IN_FEATURES,
                                 group_size=GROUP_SIZE, seed=SEED, passes=prefix)
        fresh_codes, fresh_scale = quantize(w, _codec(), prefix, group_size=GROUP_SIZE, seed=SEED)
        from_fresh = dequantize(fresh_codes, fresh_scale, _codec(), IN_FEATURES,
                                group_size=GROUP_SIZE, seed=SEED)
        assert torch.equal(from_prefix, from_fresh)
