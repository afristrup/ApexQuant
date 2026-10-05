"""Sites for a model that has no BatchNorm at all, such as ConvNeXt."""

from __future__ import annotations

from torch import nn

from .activation_kinds import ACT_UNKNOWN
from .case_letters import CASE_D
from .conv_bn_site import ConvBnSite

UNKNOWN_SOURCE = "unknown"


def unnormalized_conv_bn_sites(model: nn.Module) -> list[ConvBnSite]:
    """Mark every convolution of a BatchNorm-free model uncorrectable.

    ConvNeXt-Tiny is the negative control of section 7.3: it has zero BatchNorm
    layers, so QBN must leave it bit-identical to the plain quantized model. No
    convolution here has a following BatchNorm, so no site is correctable and no
    topology needs resolving.

    Args:
        model: Any module containing no BatchNorm.

    Returns:
        One case-D :class:`ConvBnSite` per ``nn.Conv2d``, all with ``bn`` None.
    """
    return [ConvBnSite(name, None, CASE_D, (), (), ACT_UNKNOWN, UNKNOWN_SOURCE)
            for name, module in model.named_modules()
            if isinstance(module, nn.Conv2d)]
