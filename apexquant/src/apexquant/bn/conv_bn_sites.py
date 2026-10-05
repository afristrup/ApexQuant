"""Resolve a vision model's convolutions to their BatchNorm neighbourhoods."""

from __future__ import annotations

from torch import nn

from .conv_bn_site import ConvBnSite
from .efficientnet_conv_bn_sites import efficientnet_conv_bn_sites
from .resnet_conv_bn_sites import resnet_conv_bn_sites
from .unnormalized_conv_bn_sites import unnormalized_conv_bn_sites

ADAPTERS = {
    "ResNet": resnet_conv_bn_sites,
    "EfficientNet": efficientnet_conv_bn_sites,
}


def conv_bn_sites(model: nn.Module) -> list[ConvBnSite]:
    """Return one record per convolution: following BatchNorm, case, producers.

    Section 4.4 of the QBN design document prefers a hand-written topology
    adapter to ``torch.fx`` for the falsification models, because a hand-written
    one is less likely to fail silently. It is held to that promise here: a model
    whose family is not recognised raises, and the resolved sites are checked to
    cover every ``nn.Conv2d`` in the model before they are returned.

    A model with no BatchNorm anywhere needs no topology at all and comes back as
    zero correctable sites, which is the ConvNeXt negative control.

    Args:
        model: A timm vision model in eval mode. Weights are never read, so
            ``pretrained=False`` is enough.

    Returns:
        One :class:`ConvBnSite` per ``nn.Conv2d``, in forward order.

    Raises:
        TypeError: The model has BatchNorm but no adapter, or an adapter missed
            a convolution.
    """
    normalized = any(isinstance(m, nn.modules.batchnorm._BatchNorm)
                     for m in model.modules())
    adapter = ADAPTERS.get(type(model).__name__)
    if not normalized:
        sites = unnormalized_conv_bn_sites(model)
    elif adapter is None:
        raise TypeError(f"no topology adapter for {type(model).__name__}; "
                        f"known families are {tuple(ADAPTERS)}")
    else:
        sites = adapter(model)
    resolved = {site.conv for site in sites}
    present = {name for name, m in model.named_modules() if isinstance(m, nn.Conv2d)}
    if resolved != present:
        raise TypeError(f"adapter missed {sorted(present - resolved)} and invented "
                        f"{sorted(resolved - present)}")
    return sites
