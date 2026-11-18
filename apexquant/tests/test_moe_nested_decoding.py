"""Router-driven nested decoding for stacked experts."""

import torch

from apexquant.moe import ProgressiveScalarMoE
from apexquant.vq.progressive_artifact import quantize
from apexquant.vq.scalar_codec import ScalarCodec


def _module():
    torch.manual_seed(4)
    experts, out_features, in_features = 3, 5, 16
    weight = torch.randn(experts, out_features, in_features)
    codec = ScalarCodec(d=8, bits=2)
    codes, scale = quantize(weight.reshape(experts * out_features, in_features),
                            codec, passes=3, group_size=8, seed=7)
    return ProgressiveScalarMoE(
        codes.reshape(3, experts, out_features, 2, -1),
        scale.reshape(3, experts, out_features, 2),
        torch.empty(0), codec, in_features, 8, 7), codec


def test_router_depths_follow_gate_weights():
    module, _ = _module()
    depths = module.router_passes(torch.tensor([[0.9, 0.45, 0.1]]), rho=0.5)
    assert torch.equal(depths, torch.tensor([[3, 2, 1]]))


def test_per_route_prefixes_match_independent_prefix_decodes():
    module, _ = _module()
    x = torch.randn(2, 16)
    indices = torch.tensor([[0, 1], [2, 1]])
    weights = torch.tensor([[0.8, 0.2], [0.6, 0.4]])
    depths = torch.tensor([[3, 1], [2, 1]])

    got = module(x, indices, weights, passes=depths, use_triton=False)
    expected = (
        module(x, indices[:, :1], weights[:, :1], passes=3, use_triton=False)
        + module(x, indices[:, 1:], weights[:, 1:], passes=1, use_triton=False)
    )
    expected[1] += module(x[1:2], indices[1:2, :1], weights[1:2, :1],
                           passes=2, use_triton=False)[0]
    expected[1] -= module(x[1:2], indices[1:2, :1], weights[1:2, :1],
                           passes=3, use_triton=False)[0]
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_expert_pass_vector_selects_each_expert_prefix():
    module, _ = _module()
    x = torch.randn(2, 16)
    indices = torch.tensor([[0, 1], [2, 1]])
    weights = torch.tensor([[0.8, 0.2], [0.6, 0.4]])
    expert_depths = torch.tensor([3, 1, 2])
    route_depths = torch.tensor([[3, 1], [2, 1]])

    got = module(x, indices, weights, passes=expert_depths, use_triton=False)
    expected = module(x, indices, weights, passes=route_depths, use_triton=False)
    assert torch.equal(got, expected)


def test_expert_pass_allocator_aggregates_router_traffic():
    module, _ = _module()
    indices = torch.tensor([[0, 1], [0, 2]])
    weights = torch.tensor([[0.9, 0.1], [0.8, 0.2]])

    depths = module.allocate_expert_passes(indices, weights, rho=0.5)

    assert torch.equal(depths, torch.tensor([3, 1, 1]))


def test_threshold_passes_equalizes_weighted_error():
    module, _ = _module()
    gates = torch.tensor([0.8, 0.2, 0.0])

    depths = module.threshold_passes(gates, rho=0.5, tolerance=0.1)

    assert torch.equal(depths, torch.tensor([3, 1, 1]))
    assert torch.all(gates * 0.5 ** depths <= 0.1 + 1e-6)


def test_budget_allocator_spends_extra_pass_on_important_expert():
    module, _ = _module()
    indices = torch.tensor([[0, 1], [0, 1]])
    weights = torch.tensor([[0.9, 0.1], [0.9, 0.1]])

    depths = module.allocate_expert_passes_budget(
        indices, weights, rho=0.5, bit_budget=4.06, bits_per_pass=2.03,
        max_passes=3, resolution=1000)

    assert torch.equal(depths, torch.tensor([3, 1, 1]))
