"""Reject artifacts whose guard fired on paths that apply one rotation per pass."""

from __future__ import annotations

from .quantized_weight import QuantizedWeight


def require_unguarded(qw: QuantizedWeight) -> None:
    """Raise if ``qw`` holds guard flags; the fused kernels rotate once per pass."""
    if qw.guard is not None:
        raise NotImplementedError(
            "this artifact has groups rotated twice by the Hadamard guard; "
            "decode it with dequantize_weight, the fused kernels do not support it"
        )
