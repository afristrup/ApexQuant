"""Triton kernel: one program per row of a layer normalisation."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(  # noqa: PLR0913
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    row_stride,
    n_cols,
    eps,
    BLOCK_N: tl.constexpr,
):
    """Normalise one row of ``x`` and apply the affine transform into ``y``.

    The row is loaded once into registers and reused for both the statistics and
    the affine output, so mean, variance and scale cost a single pass over
    memory.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = centered * rstd * weight + bias

    tl.store(y_ptr + row * row_stride + cols, y, mask=mask)
