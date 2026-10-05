import torch
from torch import nn

from apexquant import ApexQuantConfig, ApexQuantLinear, quantize_linear_modules


def test_turboquant_linear_approximates_dense():
    torch.manual_seed(0)
    dense = nn.Linear(256, 128)
    x = torch.randn(8, 256)
    fused = ApexQuantLinear.from_linear(dense, ApexQuantConfig(passes=2))

    rel = (fused(x) - dense(x)).norm() / dense(x).norm()
    assert rel < 0.05
    assert fused.bias is not None


def test_quantize_linear_modules_replaces_and_skips():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(256, 256),  # replaced
        nn.ReLU(),
        nn.Linear(256, 64),   # replaced (in_features >= group_size)
        nn.Linear(64, 10),    # skipped: in_features 64 < group_size 128
    )
    replaced = quantize_linear_modules(model, ApexQuantConfig())
    assert replaced == 2
    kinds = [type(m).__name__ for m in model]
    assert kinds == ["ApexQuantLinear", "ReLU", "ApexQuantLinear", "Linear"]


def test_state_dict_roundtrip():
    torch.manual_seed(0)
    fused = ApexQuantLinear.from_linear(nn.Linear(128, 32), ApexQuantConfig())
    clone = ApexQuantLinear.from_linear(nn.Linear(128, 32), ApexQuantConfig())
    clone.load_state_dict(fused.state_dict())
    x = torch.randn(4, 128)
    assert torch.allclose(fused(x), clone(x), atol=1e-5)
