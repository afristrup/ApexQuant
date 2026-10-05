"""Numeric constants of the Clay architecture.

Every magic number the original ``claymodel`` package spelled inline lives here,
so that a change to the architecture is a change to one line.
"""

from __future__ import annotations

# Width of the metadata block (normalised week, hour, lat, lon -- 4 + 4) that is
# concatenated onto every 2-D position encoding, so the sincos table is only
# ``dim - METADATA_DIM`` wide.
METADATA_DIM = 8

# Base of the geometric frequency ladder in every sincos encoding.
SINCOS_TEMPERATURE = 10000

# Width of the wavelength embedding fed to the dynamic-weight generator.
WAVE_DIM = 128

# Number of learned latent tokens the weight generator prepends to the waves.
NUM_LATENT_TOKENS = 128

# The generated convolution/linear weights are scaled down by this factor before
# use; without it the dynamically produced weights start far too large.
DYNAMIC_WEIGHT_SCALE = 0.02

# Standard deviation of the learned tokens (class token, latent wave tokens).
TOKEN_INIT_STD = 0.02

# Shape of the small transformer that turns wavelengths into patch weights.
WAVES_TRANSFORMER_HEADS = 4
WAVES_TRANSFORMER_LAYERS = 1

# ``torch.nn.LayerNorm`` default, restated so the Triton path matches it.
LAYER_NORM_EPS = 1e-5

# Set this environment variable to force every op onto its torch reference
# path. See clay.ops.use_triton for what the Triton path costs and buys.
DISABLE_TRITON_ENV = "CLAY_DISABLE_TRITON"

# Accumulation precision of ``tl.dot`` inside the kernels. "tf32" uses the
# tensor cores and is what makes the Triton path faster than cuBLAS; "ieee"
# reproduces the original exactly and is what the parity tests pin.
TRITON_DOT_PRECISION = "tf32"

# Patch size of the published v1.5 checkpoints.
DEFAULT_PATCH_SIZE = 8
