"""Checkpoint portability for a fused layer.

The pretrained-fuse-reload and CPU/CUDA-parity tests that used to live here
quantized an ``EarthObservationEncoder``, which was dropped along with the rest
of ``eo/`` (Table 4 evaluates the real Clay v1.5 checkpoint via ``clay-model``
instead of this synthetic stand-in). Only the checkpoint-metadata test survives,
since it exercises ``ApexQuantLinear`` directly and has no eo dependency.
"""

import torch
from torch import nn

from apexquant import ApexQuantConfig, ApexQuantLinear


def test_extra_state_makes_checkpoint_self_describing():
    fused = ApexQuantLinear.from_linear(nn.Linear(256, 64), ApexQuantConfig(seed=7, passes=2))
    sd = fused.state_dict()
    assert any(k.endswith("_extra_state") for k in sd)

    # A layer fused with a *different* config has matching buffer shapes but
    # different metadata; loading the checkpoint must restore the saved metadata.
    other = ApexQuantLinear.from_linear(nn.Linear(256, 64), ApexQuantConfig(seed=999, passes=2))
    other.load_state_dict(sd)
    assert other.seed == 7 and other.group_size == 128 and other.bits == 4
    x = torch.randn(4, 256)
    assert torch.allclose(fused(x), other(x), atol=1e-5)
