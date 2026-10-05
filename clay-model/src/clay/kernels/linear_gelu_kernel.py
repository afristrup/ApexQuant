"""Triton kernel: tiled GEMM with a bias + exact-GELU epilogue."""

from __future__ import annotations

import triton
import triton.language as tl

# 1 / sqrt(2), the argument scale of the exact (erf) GELU. Triton only lets a
# jitted body read a global that is already a constexpr.
INV_SQRT2 = tl.constexpr(0.7071067811865476)


# The kernel is memory-bound at small tiles and register-bound at large ones,
# and where the crossover sits depends on the layer width, so the shape is part
# of the autotune key.
CONFIGS = [
    triton.Config(
        {"BLOCK_M": m, "BLOCK_N": n, "BLOCK_K": k}, num_warps=warps, num_stages=stages
    )
    for m, n, k, warps, stages in [
        (64, 64, 32, 4, 3),
        (128, 64, 32, 4, 4),
        (64, 128, 32, 4, 4),
        (128, 128, 32, 8, 3),
        (128, 128, 64, 8, 4),
        (256, 128, 64, 8, 3),
    ]
]


@triton.autotune(configs=CONFIGS, key=["n_rows", "n_out", "n_in", "PRECISION"])
@triton.jit
def linear_gelu_kernel(  # noqa: PLR0913
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_rows,
    n_out,
    n_in,
    x_row_stride,
    weight_row_stride,
    out_row_stride,
    PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute one ``BLOCK_M x BLOCK_N`` tile of ``gelu(x @ weight.T + bias)``.

    ``weight`` is taken in ``nn.Linear`` layout ``(n_out, n_in)``, so the inner
    loop walks both operands along their contiguous axis and the transpose costs
    nothing.
    """
    tile_m = tl.program_id(0)
    tile_n = tl.program_id(1)

    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, n_in, BLOCK_K):
        k = start_k + offs_k
        k_mask = k < n_in
        x = tl.load(
            x_ptr + offs_m[:, None] * x_row_stride + k[None, :],
            mask=(offs_m[:, None] < n_rows) & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            weight_ptr + offs_n[:, None] * weight_row_stride + k[None, :],
            mask=(offs_n[:, None] < n_out) & k_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w), input_precision=PRECISION)

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < n_out, other=0.0).to(tl.float32)
    pre = acc + bias[None, :]
    activated = 0.5 * pre * (1.0 + tl.math.erf(pre * INV_SQRT2))

    tl.store(
        out_ptr + offs_m[:, None] * out_row_stride + offs_n[None, :],
        activated.to(out_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < n_rows) & (offs_n[None, :] < n_out),
    )
