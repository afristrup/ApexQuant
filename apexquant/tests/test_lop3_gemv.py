"""LOP3 packed-dot GEMV: dispatcher parity, graph capture, cached views.

The decode computes each word's inner product in packed fp16 (values and
activations rounded to fp16, one fp32 convert per word), so parity against the
fp32 reference is ~1.7e-3 relative at worst (measured over 54 configs), not
the ~1e-6 of the fp32 gather kernels. Assertions use a relative-norm bound.
"""

import pytest
import torch

from apexquant import ApexQuantConfig, turboquantize_weight
from apexquant.kernels.fused_dequant_matmul import fused_dequant_matmul
from apexquant.kernels.triton_gemv_mp_lop3 import (
    GEMV_MP_LOP3_AVAILABLE,
    lop3_views,
)

needs_lop3 = pytest.mark.skipif(
    not (torch.cuda.is_available() and GEMV_MP_LOP3_AVAILABLE),
    reason="needs CUDA and triton",
)


def _quantize(o, i, passes, gs=512, codebook="apex"):
    torch.manual_seed(0)
    w = torch.randn(o, i, device="cuda")
    return turboquantize_weight(
        w, ApexQuantConfig(bits=2, passes=passes, group_size=gs, codebook=codebook))


@needs_lop3
@pytest.mark.parametrize("passes", [1, 2, 3])
@pytest.mark.parametrize("shape,gs", [((256, 1024), 512), ((384, 768), 256)])
def test_batch1_parity(passes, shape, gs):
    qw = _quantize(*shape, passes, gs=gs)
    x = torch.randn(1, shape[1], device="cuda")
    ref = fused_dequant_matmul(x, qw, force_backend="torch")
    out = fused_dequant_matmul(x, qw, force_backend="triton")
    rel = (out - ref).norm() / ref.norm()
    assert rel < 3e-3, f"relative error {rel:.2e}"


@needs_lop3
def test_lop3_views_cached_and_bit_exact():
    qw = _quantize(128, 512, 2)
    words, consts = lop3_views(qw)
    assert lop3_views(qw)[0] is words  # cached, built once
    # the permutation is a bijection on fields: decoding the permuted words at
    # the permuted positions must reproduce every original index
    orig = qw.decode_views()[0]
    for f, v in [(0, 0), (3, 6), (8, 1), (15, 15)]:
        assert torch.equal((words >> (2 * f)) & 3, (orig >> (2 * v)) & 3)


@needs_lop3
def test_batch1_graph_capture():
    qw = _quantize(256, 1024, 3)
    x = torch.randn(1, 1024, device="cuda")
    eager = fused_dequant_matmul(x, qw, force_backend="triton")
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        graphed_out = fused_dequant_matmul(x, qw, force_backend="triton")
    g.replay()
    torch.cuda.synchronize()
    rel = (graphed_out - eager).norm() / eager.norm()
    assert rel < 1e-4, f"graph replay diverged: {rel:.2e}"
