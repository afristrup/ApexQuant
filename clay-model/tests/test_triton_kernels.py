"""Each Triton kernel must agree with the torch reference it replaces."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from clay.constants import DISABLE_TRITON_ENV, LAYER_NORM_EPS, METADATA_DIM
from clay.ops.sincos_2d_with_gsd import sincos_2d_with_gsd

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the Triton kernels need a CUDA device"
)

TOKENS = 1025
ENCODER_DIM = 1024
HEADS = 16
BATCH = 2


def test_layer_norm_matches_torch():
    from clay.kernels.triton_layer_norm import triton_layer_norm

    x = torch.randn(BATCH, TOKENS, ENCODER_DIM, device="cuda")
    weight = torch.randn(ENCODER_DIM, device="cuda")
    bias = torch.randn(ENCODER_DIM, device="cuda")

    torch.testing.assert_close(
        triton_layer_norm(x, weight, bias, LAYER_NORM_EPS),
        F.layer_norm(x, (ENCODER_DIM,), weight, bias, LAYER_NORM_EPS),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("head_dim", [64, 48])
def test_flash_attention_matches_torch(head_dim):
    """At ieee precision the kernel must reproduce torch, ragged tail included."""
    from clay.kernels.triton_flash_attention import triton_flash_attention

    shape = (BATCH, HEADS, TOKENS, head_dim)
    q, k, v = (torch.randn(shape, device="cuda") for _ in range(3))

    torch.testing.assert_close(
        triton_flash_attention(q, k, v, precision="ieee"),
        F.scaled_dot_product_attention(q, k, v),
        rtol=1e-4,
        atol=1e-4,
    )


def test_linear_gelu_matches_torch():
    """At ieee precision the fused epilogue must reproduce torch."""
    from clay.kernels.triton_linear_gelu import triton_linear_gelu

    x = torch.randn(BATCH, TOKENS, 256, device="cuda")
    weight = torch.randn(512, 256, device="cuda") / 16
    bias = torch.randn(512, device="cuda")

    torch.testing.assert_close(
        triton_linear_gelu(x, weight, bias, precision="ieee"),
        F.gelu(F.linear(x, weight, bias)),
        rtol=1e-4,
        atol=1e-4,
    )


def test_default_precision_is_tf32_and_stays_within_its_error_budget():
    """The shipped default trades mantissa bits for tensor cores, but bounded.

    TF32 keeps 10 mantissa bits against fp32's 23, so a single matmul drifts by
    roughly 1e-3 relative. Anything materially worse than that would mean a bug
    in the kernel rather than the expected cost of the format.
    """
    from clay.kernels.triton_flash_attention import triton_flash_attention

    shape = (BATCH, HEADS, TOKENS, 64)
    q, k, v = (torch.randn(shape, device="cuda") for _ in range(3))

    torch.testing.assert_close(
        triton_flash_attention(q, k, v),
        F.scaled_dot_product_attention(q, k, v),
        rtol=5e-3,
        atol=5e-3,
    )


def test_sincos_2d_matches_torch(monkeypatch):
    from clay.kernels.triton_sincos_2d_with_gsd import triton_sincos_2d_with_gsd

    grid = 32
    dim = ENCODER_DIM - METADATA_DIM
    gsd = torch.tensor(10.0)
    device = torch.device("cuda")

    monkeypatch.setenv(DISABLE_TRITON_ENV, "1")
    reference = sincos_2d_with_gsd(grid, grid, dim, gsd, device)
    monkeypatch.delenv(DISABLE_TRITON_ENV)

    torch.testing.assert_close(
        triton_sincos_2d_with_gsd(grid, grid, dim, gsd, 10000, device),
        reference,
        rtol=1e-5,
        atol=1e-6,
    )
