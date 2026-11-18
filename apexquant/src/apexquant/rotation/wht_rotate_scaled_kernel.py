"""Triton kernel: per-column scale, then each pass's randomized Hadamard, in one load."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def wht_rotate_scaled_kernel(  # noqa: PLR0913
    x_ptr,
    col_ptr,
    sign_ptr,
    out_ptr,
    n_rows,
    n_groups,
    rows_per_pass,
    scale,
    D: tl.constexpr,
    LOG_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """``wht_rotate_kernel`` with the input multiplied by a per-input-column scale.

    Rows are groups of ``D`` consecutive input columns, ``n_groups`` per token, so
    row ``r`` holds columns ``(r % n_groups) * D + i``. The column scale is fetched
    with the same mask as the input and folded into the sign multiply, so the
    Sinkhorn column diagonal costs no extra pass over the activations.
    """
    pass_id = tl.program_id(1)
    row = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    col = tl.arange(0, D)
    keep = row < n_rows

    v = tl.load(x_ptr + row[:, None] * D + col[None, :], mask=keep[:, None], other=0.0)
    group = row % n_groups
    c = tl.load(col_ptr + group[:, None] * D + col[None, :], mask=keep[:, None], other=0.0)
    v = v * c * tl.load(sign_ptr + pass_id * D + col)[None, :]

    for _ in range(LOG_D):
        a, b = tl.split(tl.reshape(v, (BLOCK_R, D // 2, 2)))
        v = tl.reshape(tl.join(a + b, a - b), (BLOCK_R, D))
        v = tl.reshape(tl.trans(tl.reshape(v, (BLOCK_R, 2, D // 2)), 0, 2, 1),
                       (BLOCK_R, D))

    tl.store(out_ptr + pass_id * rows_per_pass + row[:, None] * D + col[None, :],
             v * scale, mask=keep[:, None])
