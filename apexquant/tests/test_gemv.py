import time

import pytest
import torch
from torch import nn

from apexquant import ApexQuantConfig, turboquantize_weight
from apexquant.kernels.fused_dequant_matmul import fused_dequant_matmul


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("d", [1024, 4096, 8192])
def test_gemv_path_matches_reference(d):
    # At M==1 the dispatcher uses the mat-vec kernel; it must match the torch oracle.
    torch.manual_seed(0)
    w = torch.randn(d, d, device="cuda")
    qw = turboquantize_weight(w, ApexQuantConfig(bits=2, passes=2, group_size=128))
    x = torch.randn(1, d, device="cuda")
    got = fused_dequant_matmul(x, qw)                       # -> GEMV
    ref = fused_dequant_matmul(x, qw, force_backend="torch")
    rel = (got - ref).norm() / ref.norm()
    assert rel < 5e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gemv_beats_dense_bf16_on_wide_layer():
    # The whole point: at batch 1 on a layer whose bf16 weight spills L2, the 2-bit
    # mat-vec moves 8x fewer bytes and beats a dense bf16 GEMV. d=8192 -> 134 MB
    # bf16 weight, well over the RTX 4090's ~72 MB L2.
    torch.manual_seed(0)
    d = 8192
    w = torch.randn(d, d, device="cuda")
    qw = turboquantize_weight(w, ApexQuantConfig(bits=2, passes=1, group_size=128))
    x = torch.randn(1, d, device="cuda")
    lin = nn.Linear(d, d, bias=False).to("cuda").half().eval()
    x16 = torch.randn(1, d, device="cuda", dtype=torch.float16)

    def clock(fn, it=100):
        for _ in range(30):
            fn()
        torch.cuda.synchronize()
        s = time.perf_counter()
        for _ in range(it):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - s) / it

    fused = clock(lambda: fused_dequant_matmul(x, qw))
    dense = clock(lambda: lin(x16))
    assert fused < dense, f"fused 2-bit {fused*1e6:.0f}us should beat dense bf16 {dense*1e6:.0f}us"
