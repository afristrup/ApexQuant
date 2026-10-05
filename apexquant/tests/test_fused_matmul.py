import pytest
import torch
import torch.nn.functional as F

from apexquant import (
    ApexQuantConfig,
    dequantize_weight,
    fused_dequant_matmul,
    turboquantize_weight,
)
from apexquant.rotation import random_rotation


def _emulate_triton_addressing(x, qw):
    """Replicate the Triton kernel's index math on CPU (no GPU needed).

    Mirrors triton_fused_dequant_matmul exactly: 2-D packed with byte column
    ``k // 2`` (stride 1), nibble picked by ``k & 1``, group ``k // gs``. If the
    kernel's addressing (e.g. the packed stride) is wrong, this diverges from the
    torch reference below.
    """
    gs = qw.group_size
    x2 = x.reshape(-1, x.shape[-1])
    pad = qw.num_groups * gs - x2.shape[-1]
    xg = F.pad(x2, (0, pad)).view(-1, qw.num_groups, gs)
    m, k_total = x2.shape[0], qw.num_groups * gs

    per_byte = 8 // qw.bits
    mask = (1 << qw.bits) - 1
    out = torch.zeros(m, qw.out_features)
    for p in range(qw.passes):
        rx = random_rotation(xg, qw.seed + p).reshape(m, k_total)  # (M, K)
        packed_2d = qw.packed[p].reshape(qw.out_features, -1)      # (N, K*bits//8)
        for k in range(k_total):
            byte = packed_2d[:, k // per_byte].int()
            idx = (byte >> (qw.bits * (k % per_byte))) & mask
            w_col = qw.levels[idx] * qw.scale[p][:, k // gs]       # (N,)
            out += rx[:, k].unsqueeze(1) * w_col.unsqueeze(0)
    return out


def test_fused_matches_dequantized_matmul():
    torch.manual_seed(0)
    w = torch.randn(128, 256)
    x = torch.randn(10, 256)
    qw = turboquantize_weight(w, ApexQuantConfig(passes=2))

    fused = fused_dequant_matmul(x, qw, force_backend="torch")
    reference = x @ dequantize_weight(qw).T
    assert torch.allclose(fused, reference, atol=1e-3, rtol=1e-3)


def test_triton_addressing_math_matches_torch():
    # Emulate the Triton kernel's exact index arithmetic on CPU; if the packed
    # stride / nibble / group mapping is wrong it diverges from the torch path.
    torch.manual_seed(0)
    w = torch.randn(96, 300)   # 300 non-multiple of 128 also exercises padding
    x = torch.randn(7, 300)
    qw = turboquantize_weight(w, ApexQuantConfig(passes=2))
    emulated = _emulate_triton_addressing(x, qw)
    reference = fused_dequant_matmul(x, qw, force_backend="torch")
    assert torch.allclose(emulated, reference, atol=1e-3, rtol=1e-3)


def test_fused_matches_dequantized_matmul_non_multiple_indim():
    # The fused == x @ dequantize_weight(qw).T identity must hold even when
    # in_features is not a multiple of group_size (padded coords are zeroed in x).
    torch.manual_seed(0)
    w = torch.randn(64, 300)
    x = torch.randn(5, 300)
    qw = turboquantize_weight(w, ApexQuantConfig(passes=2))
    fused = fused_dequant_matmul(x, qw, force_backend="torch")
    reference = x @ dequantize_weight(qw).T
    assert torch.allclose(fused, reference, atol=1e-3, rtol=1e-3)


def test_fused_preserves_leading_dims():
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    x = torch.randn(2, 5, 128)
    qw = turboquantize_weight(w, ApexQuantConfig())
    out = fused_dequant_matmul(x, qw, force_backend="torch")
    assert out.shape == (2, 5, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("group_size", [128, 256, 512])
def test_triton_wide_group_fits_shared_memory(group_size):
    # A group_size wider than the K-tile once blew the shared-memory budget on
    # the GPU kernel (group_size=256 -> ~192 KB > ~99 KB). BLOCK_K is now capped
    # independently of the group, so wide groups run and still match the oracle.
    torch.manual_seed(0)
    w = torch.randn(256, 1024, device="cuda")
    x = torch.randn(8, 1024, device="cuda")
    qw = turboquantize_weight(w, ApexQuantConfig(bits=2, passes=2, group_size=group_size))
    triton_out = fused_dequant_matmul(x, qw)                       # CUDA -> Triton
    torch_out = fused_dequant_matmul(x, qw, force_backend="torch")
    rel = (triton_out - torch_out).norm() / torch_out.norm()
    assert rel < 5e-3
