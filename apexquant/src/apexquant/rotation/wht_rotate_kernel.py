"""Triton kernel: sign-flip and Walsh-Hadamard a tile of grouped rows."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def wht_rotate_kernel(  # noqa: PLR0913
    x_ptr,
    sign_ptr,
    out_ptr,
    n_rows,
    rows_per_pass,
    scale,
    D: tl.constexpr,
    LOG_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Apply pass ``p``'s randomized Hadamard rotation to a tile of rows.

    Each stage butterflies *adjacent* pairs and then perfect-shuffles, so both
    shapes it needs are the same on every stage and nothing in the body depends
    on the loop index. That matters for more than tidiness: a Triton loop
    variable is a tensor, so it cannot size a reshape, and the natural
    formulation -- stage ``j`` pairs across stride ``2**j`` -- will not compile.
    Over ``LOG_D`` stages the shuffles compose back to the identity while each
    round butterflies a different bit, so this is the natural-order transform and
    equals :func:`walsh_hadamard_transform` rather than a permutation of it.
    """
    pass_id = tl.program_id(1)
    row = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    col = tl.arange(0, D)
    keep = row < n_rows

    v = tl.load(x_ptr + row[:, None] * D + col[None, :],
                mask=keep[:, None], other=0.0)
    v = v * tl.load(sign_ptr + pass_id * D + col)[None, :]

    for _ in range(LOG_D):
        a, b = tl.split(tl.reshape(v, (BLOCK_R, D // 2, 2)))
        v = tl.reshape(tl.join(a + b, a - b), (BLOCK_R, D))
        v = tl.reshape(tl.trans(tl.reshape(v, (BLOCK_R, 2, D // 2)), 0, 2, 1),
                       (BLOCK_R, D))

    tl.store(out_ptr + pass_id * rows_per_pass + row[:, None] * D + col[None, :],
             v * scale, mask=keep[:, None])
