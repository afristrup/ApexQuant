"""Quantize a vision model in place and hand back the weight error per layer."""

from __future__ import annotations

import torch
from torch import nn

from ..config import ApexQuantConfig
from ..quantize.dequantize_weight import dequantize_weight
from ..quantize.turboquantize_weight import turboquantize_weight
from .harness_group_size import harness_group_size
from .layer_residual import LayerResidual

#: The vision harness's settings, fixed: 2-bit indices on the density-matched
#: ApexQuant codebook.
BITS = 2
CODEBOOK = "apex"
MATRIX_NDIM = (2, 4)


def quantize_with_residuals(model: nn.Module, passes: int) -> dict[str, LayerResidual]:
    """Replace every eligible weight by its ``passes``-pass reconstruction, keeping ``dW``.

    Section 4.3 needs ``dW = W_hat - W``, which
    ``experiments/progressive_vision_imagenet.py::quantize_in_place`` computes and
    throws away. That function is what the paper's accuracy numbers come from, so
    it is left exactly as it is and this is a separate entry point with the same
    selection rule, the same reshape and the same configuration.

    Peak extra memory is one fp32 copy of the quantized layers, 25M floats for
    ResNet-50, which is nothing.

    Order matters and is forced in both directions (section 4.2). The correction
    needs ``dW``, which only exists after quantization, and it needs the
    *unfolded* ``(gamma, beta)`` of the preceding BatchNorm, which folding
    destroys. So: quantize, then :func:`apply_qbn`, then fold if you want to.

    Args:
        model: The model to quantize, in place, on CPU or GPU.
        passes: Residual passes ``P``.

    Returns:
        Dict from module name to its :class:`LayerResidual`. Layers the group
        rule leaves dense do not appear.
    """
    residuals: dict[str, LayerResidual] = {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            weight = getattr(mod, "weight", None)
            if weight is None or weight.ndim not in MATRIX_NDIM:
                continue
            flat = weight.reshape(weight.shape[0], -1)
            gs = harness_group_size(flat.shape[1])
            if gs is None:
                continue
            config = ApexQuantConfig(bits=BITS, passes=passes, group_size=gs,
                                     codebook=CODEBOOK)
            reconstruction = dequantize_weight(turboquantize_weight(flat.float(), config))
            reconstruction = reconstruction.reshape(weight.shape).to(weight.dtype)
            residuals[name] = LayerResidual(dw=(reconstruction - weight).clone(),
                                            group_size=gs)
            weight.copy_(reconstruction)
    return residuals
