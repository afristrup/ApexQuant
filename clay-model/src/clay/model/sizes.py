"""Encoder and decoder shapes of the four published Clay model sizes."""

from __future__ import annotations

# Each entry is (dim, depth, heads, dim_head, mlp_ratio) for the encoder and for
# the decoder. The decoder stays small at every size: it exists only to make the
# masked-autoencoder objective well posed and is discarded at inference.
SIZES = {
    "tiny": {
        "encoder": {"dim": 192, "depth": 6, "heads": 4, "dim_head": 48, "mlp_ratio": 2},
        "decoder": {"dim": 96, "depth": 3, "heads": 2, "dim_head": 48, "mlp_ratio": 2},
    },
    "small": {
        "encoder": {"dim": 384, "depth": 6, "heads": 6, "dim_head": 64, "mlp_ratio": 2},
        "decoder": {"dim": 192, "depth": 4, "heads": 4, "dim_head": 64, "mlp_ratio": 2},
    },
    "base": {
        "encoder": {
            "dim": 768,
            "depth": 12,
            "heads": 12,
            "dim_head": 64,
            "mlp_ratio": 4,
        },
        "decoder": {"dim": 512, "depth": 4, "heads": 4, "dim_head": 64, "mlp_ratio": 4},
    },
    "large": {
        "encoder": {
            "dim": 1024,
            "depth": 24,
            "heads": 16,
            "dim_head": 64,
            "mlp_ratio": 4,
        },
        "decoder": {"dim": 512, "depth": 4, "heads": 4, "dim_head": 64, "mlp_ratio": 4},
    },
}
