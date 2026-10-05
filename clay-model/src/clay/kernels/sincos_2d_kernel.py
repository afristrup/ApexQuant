"""Triton kernel: build one row of the GSD-scaled 2-D sincos position table."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def sincos_2d_kernel(
    out_ptr,
    omega_ptr,
    grid_width,
    quarter,
    row_stride,
    BLOCK_Q: tl.constexpr,
):
    """Write the four sin/cos quarters of position ``l`` into ``out``.

    One program owns one patch position, so the whole table is produced in a
    single launch with no meshgrid, no broadcasting of the frequency ladder and
    no concatenation of four intermediates.
    """
    position = tl.program_id(0)
    row = position // grid_width
    col = position % grid_width

    offs = tl.arange(0, BLOCK_Q)
    mask = offs < quarter
    omega = tl.load(omega_ptr + offs, mask=mask, other=0.0)

    col_angle = col.to(tl.float32) * omega
    row_angle = row.to(tl.float32) * omega

    base = out_ptr + position * row_stride
    tl.store(base + offs, tl.sin(col_angle), mask=mask)
    tl.store(base + quarter + offs, tl.cos(col_angle), mask=mask)
    tl.store(base + 2 * quarter + offs, tl.sin(row_angle), mask=mask)
    tl.store(base + 3 * quarter + offs, tl.cos(row_angle), mask=mask)
