"""Names for the activation applied between a producer BatchNorm and its consumer."""

from __future__ import annotations

ACT_IDENTITY = "identity"
ACT_RELU = "relu"
ACT_RELU6 = "relu6"
ACT_SILU = "silu"
ACT_GELU = "gelu"
ACT_SIGMOID = "sigmoid"
ACT_HARDSWISH = "hardswish"
ACT_HARDSIGMOID = "hardsigmoid"
ACT_IMAGE = "image"          # not an activation: the dataset normalization
ACT_UNKNOWN = "unknown"
