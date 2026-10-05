"""Triton kernel: streaming (FlashAttention-style) scaled dot-product attention."""

from __future__ import annotations

import triton
import triton.language as tl


# Sequence length decides whether the inner loop is worth deep pipelining, and
# head width decides the register pressure, so both are autotune keys. The tile
# heights matter more than they look: Clay's 785 tokens over 16 heads give 13
# query tiles at BLOCK_M=64, which is 208 programs against 128 SMs and so a
# lopsided second wave, while 32 spreads into 400 and 128 leaves the device
# short. Let the autotuner settle it per shape.
CONFIGS = [
    triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=warps, num_stages=stages)
    for m, n, warps, stages in [
        (32, 64, 4, 2), (32, 64, 4, 3), (32, 128, 4, 3),
        (64, 32, 4, 3), (64, 64, 4, 2), (64, 64, 4, 3), (64, 64, 8, 3),
        (64, 128, 4, 3), (64, 128, 4, 4), (64, 128, 8, 4),
        (128, 32, 4, 3), (128, 64, 4, 3), (128, 64, 8, 4),
        (128, 128, 8, 3), (128, 128, 8, 4),
    ]
]

# log2(e). The online softmax is carried in base two so the exponential is one
# hardware instruction: exp2 is an SFU op, while exp is a multiply by this
# constant followed by the same exp2. Folding it into the score scale moves that
# multiply out of the inner loop, where it ran once per score, to once per call.
LOG2_E = tl.constexpr(1.4426950408889634)


@triton.autotune(configs=CONFIGS, key=["n_tokens", "head_dim", "PRECISION"])
@triton.jit
def flash_attention_kernel(  # noqa: PLR0913
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    softmax_scale,
    batch_stride,
    token_stride,
    n_tokens,
    head_dim,
    PRECISION: tl.constexpr,
    EVEN_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Attend one ``BLOCK_M`` query tile of one (batch, head) against all keys.

    The softmax is accumulated online -- running maximum, running denominator,
    rescaled accumulator -- so the ``n_tokens x n_tokens`` score matrix is never
    written to memory. Clay's encoder is bidirectional and dropout-free at
    inference, so there is no mask to apply beyond the ragged tail of the
    sequence.
    """
    tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    base = batch_head * batch_stride

    offs_m = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    row_mask = offs_m < n_tokens
    # Every Clay head is 64 wide, a power of two, so the head axis needs no mask
    # at all. Saying so as a constexpr lets the loads drop it rather than AND a
    # known-true predicate into every one of them.
    dim_mask = offs_d < head_dim if not EVEN_D else offs_d >= 0

    q = tl.load(
        q_ptr + base + offs_m[:, None] * token_stride + offs_d[None, :],
        mask=row_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    running_max = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, n_tokens, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        col_mask = offs_n < n_tokens
        k = tl.load(
            k_ptr + base + offs_n[:, None] * token_stride + offs_d[None, :],
            mask=col_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        # softmax_scale already carries log2(e), so scores are in base-two
        # units and both exponentials below are a single exp2.
        scores = tl.dot(q, tl.trans(k), input_precision=PRECISION) * softmax_scale
        scores = tl.where(col_mask[None, :], scores, float("-inf"))

        tile_max = tl.maximum(running_max, tl.max(scores, axis=1))
        rescale = tl.math.exp2(running_max - tile_max)
        probs = tl.math.exp2(scores - tile_max[:, None])

        running_sum = running_sum * rescale + tl.sum(probs, axis=1)
        acc = acc * rescale[:, None]

        v = tl.load(
            v_ptr + base + offs_n[:, None] * token_stride + offs_d[None, :],
            mask=col_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(probs.to(v.dtype), v, input_precision=PRECISION)
        running_max = tile_max

    acc = acc / running_sum[:, None]
    tl.store(
        out_ptr + base + offs_m[:, None] * token_stride + offs_d[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=row_mask[:, None] & dim_mask[None, :],
    )
