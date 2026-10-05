"""Regression tests for stacked-expert (MoE) tensor layouts.

These exist because of a silent failure, not a hypothetical one. Requiring a 2D
shape meant a fused MoE checkpoint had its routed experts skipped entirely: on
Qwen3.5-35B-A3B that is 89.6% of the parameters, and what got quantized instead
was the multi-token-prediction head. Nothing raised. The bits-per-weight came out
exactly as claimed, because it was measured over the 6.3% that had been touched.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "experiments"))

from quantise_hf_progressive import is_quantisable  # noqa: E402

from apexquant import (  # noqa: E402
    ApexQuantConfig,
    dequantize_weight,
    turboquantize_weight,
)

# (name, shape, expected) drawn from the real Qwen3.5-35B-A3B weight index.
QWEN35_CASES = [
    ("model.language_model.layers.0.mlp.experts.gate_up_proj", (256, 1024, 2048), True),
    ("model.language_model.layers.0.mlp.experts.down_proj", (256, 2048, 512), True),
    ("model.language_model.layers.0.self_attn.q_proj.weight", (8192, 2048), True),
    ("model.language_model.layers.0.linear_attn.out_proj.weight", (2048, 4096), True),
    ("model.language_model.layers.0.mlp.shared_expert.down_proj.weight", (2048, 512), True),
    # A width-4 causal convolution is 3D but its fan-in is 4, which is the
    # regime the density-matched codebook degenerates in. It must stay dense.
    ("model.language_model.layers.0.linear_attn.conv1d.weight", (8192, 1, 4), False),
    ("model.language_model.layers.0.linear_attn.A_log", (8192,), False),
    ("model.language_model.layers.0.input_layernorm.weight", (2048,), False),
    ("model.language_model.embed_tokens.weight", (248320, 2048), False),
    ("lm_head.weight", (248320, 2048), False),
]


def test_qwen35_tensor_layouts_are_classified_correctly():
    for name, shape, expected in QWEN35_CASES:
        assert is_quantisable(name, shape) is expected, f"{name} {shape}"


def test_stacked_experts_are_not_skipped_for_lacking_a_weight_suffix():
    # The 2D rule required a ".weight" suffix as well as ndim == 2. Fused expert
    # tensors have neither, so either check alone would have skipped them.
    name = "model.language_model.layers.7.mlp.experts.gate_up_proj"
    assert not name.endswith(".weight")
    assert is_quantisable(name, (256, 1024, 2048))


def test_the_head_is_still_opt_in_and_the_conv_never_is():
    head = ("lm_head.weight", (248320, 2048))
    assert not is_quantisable(*head)
    assert is_quantisable(*head, quantize_head=True)
    # No flag turns a width-4 convolution into a quantization target.
    conv = ("model.language_model.layers.0.linear_attn.conv1d.weight", (8192, 1, 4))
    assert not is_quantisable(*conv)
    assert not is_quantisable(*conv, quantize_head=True)


def test_exclude_regex_reaches_stacked_experts():
    name = "model.language_model.layers.3.linear_attn.out_proj.weight"
    import re
    assert is_quantisable(name, (2048, 4096))
    assert not is_quantisable(name, (2048, 4096), re.compile("linear_attn.out_proj"))
    stacked = "model.language_model.layers.3.mlp.experts.down_proj"
    assert not is_quantisable(stacked, (256, 2048, 512), re.compile("mlp.experts"))


def test_expert_axis_flattening_keeps_experts_independent():
    # (E, out, in) -> (E*out, in) is only correct if the contraction stays the
    # last axis, so each expert's rows are quantized against their own groups
    # and no group ever straddles two experts.
    torch.manual_seed(0)
    E, out, in_f = 4, 8, 256
    w = torch.randn(E, out, in_f)
    # Make experts wildly different in scale; a flattening that mixed them would
    # let one expert's norm set another's quantization grid.
    w = w * torch.tensor([1.0, 100.0, 0.01, 10.0]).view(E, 1, 1)

    cfg = ApexQuantConfig(bits=4, group_size=in_f, passes=2, codebook="apex")
    flat = w.reshape(E * out, in_f)
    recon = dequantize_weight(turboquantize_weight(flat, cfg)).reshape(E, out, in_f)

    assert recon.shape == w.shape
    # Every expert must be reconstructed to comparable relative accuracy despite
    # the 10,000x spread in absolute scale.
    rel = [((recon[e] - w[e]).norm() / w[e].norm()).item() for e in range(E)]
    assert max(rel) < 0.05, rel
    assert max(rel) / min(rel) < 3.0, rel


def test_flattening_round_trips_shape_for_both_expert_projections():
    torch.manual_seed(0)
    cfg = ApexQuantConfig(bits=2, group_size=256, passes=1, codebook="apex")
    for shape in ((8, 64, 512), (8, 128, 256)):
        E, out, in_f = shape
        w = torch.randn(*shape)
        recon = dequantize_weight(
            turboquantize_weight(w.reshape(E * out, in_f), cfg)
        ).reshape(E, out, in_f)
        assert recon.shape == w.shape
        assert torch.isfinite(recon).all()
