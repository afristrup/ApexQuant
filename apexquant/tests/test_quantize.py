import torch

from apexquant import ApexQuantConfig, dequantize_weight, turboquantize_weight
from apexquant.quantize import pack_int4, unpack_int4


def test_pack_unpack_roundtrip():
    idx = torch.randint(0, 16, (3, 4, 128), dtype=torch.uint8)
    assert torch.equal(unpack_int4(pack_int4(idx)), idx)


def test_weight_roundtrip_low_error():
    torch.manual_seed(0)
    w = torch.randn(256, 512)
    qw = turboquantize_weight(w, ApexQuantConfig(passes=1))
    rel = (dequantize_weight(qw) - w).norm() / w.norm()
    assert rel < 0.10  # single-pass 4-bit


def test_residual_pass_reduces_error():
    torch.manual_seed(0)
    w = torch.randn(256, 512)
    rel1 = (dequantize_weight(turboquantize_weight(w, ApexQuantConfig(passes=1))) - w).norm() / w.norm()
    rel2 = (dequantize_weight(turboquantize_weight(w, ApexQuantConfig(passes=2))) - w).norm() / w.norm()
    assert rel2 < rel1


def test_non_multiple_input_dim_is_padded():
    torch.manual_seed(0)
    w = torch.randn(32, 300)  # 300 is not a multiple of 128
    qw = turboquantize_weight(w, ApexQuantConfig())
    recon = dequantize_weight(qw)
    assert recon.shape == w.shape
    assert (recon - w).norm() / w.norm() < 0.12
