import pytest
import torch

from apexquant.config import ApexQuantConfig
from apexquant.kernels.fused_dequant_matmul import fused_dequant_matmul
from apexquant.quantize.dequantize_weight import dequantize_weight
from apexquant.quantize.turboquantize_weight import turboquantize_weight
from apexquant.rotation.guarded_rotation import guarded_rotation
from apexquant.rotation.inverse_guarded_rotation import inverse_guarded_rotation
from apexquant.rotation.normalized_third_moment import normalized_third_moment
from apexquant.rotation.random_rotation import random_rotation

D = 512
THRESHOLD = 4.0


def _spiky_weight():
    torch.manual_seed(0)
    weight = torch.randn(8, 2 * D)
    weight[0, :D] = 0
    weight[0, 7] = 3.0
    return weight


def _config(passes, threshold):
    return ApexQuantConfig(group_size=D, passes=passes, bits=2, codebook="apex", guard_threshold=threshold)


def test_third_moment_reference_values():
    torch.manual_seed(0)
    axis = torch.zeros(D)
    axis[3] = 2.0
    assert normalized_third_moment(axis).item() == pytest.approx(D ** 0.5, rel=1e-5)
    assert normalized_third_moment(torch.ones(D)).item() == pytest.approx(1.0, rel=1e-5)
    assert 1.4 < normalized_third_moment(torch.randn(4096, D)).mean().item() < 1.8
    assert normalized_third_moment(torch.zeros(D)).item() == 0


def test_guard_flags_only_spiky_groups_and_inverts():
    x = _spiky_weight().view(8, 2, D)
    rotated, flags = guarded_rotation(x, 5, THRESHOLD)
    assert flags.sum().item() == 1 and flags[0, 0]
    assert torch.allclose(rotated[1:], random_rotation(x, 5)[1:], atol=1e-6)
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-4)
    assert torch.allclose(inverse_guarded_rotation(rotated, 5, flags), x, atol=1e-4)


def test_second_rotation_delocalizes_an_axis():
    axis = torch.zeros(1, D)
    axis[0, 7] = 1.0
    once = random_rotation(axis, 5)
    twice, _ = guarded_rotation(axis, 5, THRESHOLD)
    assert once.abs().max().item() == pytest.approx(D ** -0.5, rel=1e-4)
    assert once.abs().min().item() == pytest.approx(D ** -0.5, rel=1e-4)
    assert twice.abs().std().item() > 0.3 * D ** -0.5


def test_guard_restores_decay_on_an_axis():
    weight = torch.zeros(1, D)
    weight[0, 7] = 1.0
    plain = dequantize_weight(turboquantize_weight(weight, _config(3, None)))
    guarded = dequantize_weight(turboquantize_weight(weight, _config(3, THRESHOLD)))
    plain_err = (weight - plain).pow(2).sum().item()
    guarded_err = (weight - guarded).pow(2).sum().item()
    assert plain_err > 0.2 ** 3
    assert guarded_err < 0.5 * plain_err


def test_unfired_guard_matches_unguarded_artifact():
    torch.manual_seed(1)
    weight = torch.randn(8, 2 * D)
    plain = turboquantize_weight(weight, _config(2, None))
    guarded = turboquantize_weight(weight, _config(2, THRESHOLD))
    assert guarded.guard is None
    assert torch.equal(plain.packed, guarded.packed)
    assert torch.equal(plain.scale, guarded.scale)


def test_guarded_prefix_identity():
    weight = _spiky_weight()
    deep = turboquantize_weight(weight, _config(3, THRESHOLD))
    shallow = turboquantize_weight(weight, _config(2, THRESHOLD))
    assert deep.guard is not None and deep.guard[0, 0, 0]
    assert torch.equal(deep.packed[:2], shallow.packed)
    assert torch.equal(deep.scale[:2], shallow.scale)
    assert torch.equal(deep.guard[:2], shallow.guard)


def test_fused_path_rejects_fired_guard():
    qw = turboquantize_weight(_spiky_weight(), _config(1, THRESHOLD))
    with pytest.raises(NotImplementedError):
        fused_dequant_matmul(torch.randn(2, 2 * D), qw)


def test_config_rejects_nonpositive_threshold():
    with pytest.raises(ValueError):
        _config(1, 0.0)
