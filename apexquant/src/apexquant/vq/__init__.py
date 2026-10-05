"""Vector/trellis quantization: the E8 and trellis stages, and the progressive
(nested-pass) mechanism that chains stage codecs into a residual ladder."""

from .e8_lattice_quantize import e8_lattice_quantize
from .e8p_codec import E8PCodec, E8P_DIM, E8P_SOURCE_SIZE, build_full_codebook, d8h_abs_source, parity_of
from .progressive_artifact import (
    DEFAULT_GROUP_SIZE,
    DEFAULT_SEED,
    dequantize,
    effective_bits,
    group_size_for,
    quantize,
    stored_bits_per_weight,
)
from .progressive_trellis_linear import ProgressiveTrellisLinear
from .scalar_codec import ScalarCodec, pack_indices, unpack_indices
from .trellis_codec import DEFAULT_TRELLIS_BITS, TrellisCodec

__all__ = [
    "DEFAULT_GROUP_SIZE",
    "DEFAULT_SEED",
    "DEFAULT_TRELLIS_BITS",
    "E8PCodec",
    "E8P_DIM",
    "E8P_SOURCE_SIZE",
    "ProgressiveTrellisLinear",
    "ScalarCodec",
    "TrellisCodec",
    "build_full_codebook",
    "d8h_abs_source",
    "dequantize",
    "e8_lattice_quantize",
    "effective_bits",
    "group_size_for",
    "pack_indices",
    "parity_of",
    "quantize",
    "stored_bits_per_weight",
    "unpack_indices",
]
