"""Hand-written topology adapter for timm's EfficientNet family."""

from __future__ import annotations

from torch import nn

from .activation_kind import activation_kind
from .activation_kinds import ACT_IDENTITY, ACT_IMAGE
from .bundled_activation import bundled_activation
from .case_letters import CASE_A, CASE_C, CASE_D
from .conv_bn_site import ConvBnSite
from .input_source import InputSource
from .residual_input_source import residual_input_source
from .site_from_source import site_from_source

IMAGE_SOURCE = "input"
BODY_OF_BLOCK = {
    "DepthwiseSeparableConv": (("conv_dw", "bn1"), ("conv_pw", "bn2")),
    "InvertedResidual": (("conv_pw", "bn1"), ("conv_dw", "bn2"),
                         ("conv_pwl", "bn3")),
}
_PASSTHROUGH_ATTRS = ("aa", "conv_s2d", "drop_path")


def efficientnet_conv_bn_sites(model: nn.Module) -> list[ConvBnSite]:
    """Resolve every convolution of an EfficientNet to its BatchNorm neighbourhood.

    Every normalization here is a ``BatchNormAct2d`` carrying the block's SiLU,
    so the activation is read off the BatchNorm rather than assumed. The squeeze
    excitation gate multiplies the activation by a data-dependent sigmoid, which
    section 4.4 puts in case D: the gate's own two convolutions and the block's
    projection convolution that reads the gated tensor are all uncorrectable.

    Args:
        model: A timm ``EfficientNet`` in eval mode. Weights are never read.

    Returns:
        One :class:`ConvBnSite` per ``nn.Conv2d``, in forward order.
    """
    sites = [ConvBnSite("conv_stem", "bn1", CASE_C, (), (), ACT_IMAGE, IMAGE_SOURCE)]
    source = InputSource(CASE_A, ("bn1",), (), bundled_activation(model.bn1), "bn1")
    for stage_index, stage in enumerate(model.blocks):
        for block_index, block in enumerate(stage):
            prefix = f"blocks.{stage_index}.{block_index}"
            source = _block_sites(sites, prefix, block, source)
    sites.append(site_from_source("conv_head", "bn2", source))
    return sites


def _block_sites(sites: list[ConvBnSite], prefix: str, block: nn.Module,
                 source: InputSource) -> InputSource:
    """Append one inverted-residual block's sites and return its output description."""
    for attr in _PASSTHROUGH_ATTRS:
        module = getattr(block, attr, None)
        if module is not None and not isinstance(module, nn.Identity):
            raise TypeError(f"{prefix}.{attr} is {type(module).__name__}, not a "
                            f"pass-through; this adapter does not model it")
    body = BODY_OF_BLOCK.get(type(block).__name__)
    if body is None:
        raise TypeError(f"{prefix} is a {type(block).__name__}, which this "
                        f"adapter does not model")
    shortcut = source
    for position, (conv, bn) in enumerate(body):
        if position == len(body) - 1:
            source = _squeeze_excite_sites(sites, prefix, block.se, source)
        sites.append(site_from_source(f"{prefix}.{conv}", f"{prefix}.{bn}", source))
        source = InputSource(CASE_A, (f"{prefix}.{bn}",), (),
                             bundled_activation(getattr(block, bn)), f"{prefix}.{bn}")
    if not block.has_skip:
        return source
    return residual_input_source(source, shortcut, ACT_IDENTITY, prefix)


def _squeeze_excite_sites(sites: list[ConvBnSite], prefix: str, se: nn.Module | None,
                          source: InputSource) -> InputSource:
    """Append the gate's two convolutions and return the gated-tensor description."""
    if se is None or isinstance(se, nn.Identity):
        return source
    reduce_name = f"{prefix}.se.conv_reduce"
    sites.append(ConvBnSite(reduce_name, None, CASE_D, (), (), source.activation,
                            source.node))
    sites.append(ConvBnSite(f"{prefix}.se.conv_expand", None, CASE_D, (), (),
                            activation_kind(se.act1), reduce_name))
    return InputSource(CASE_D, (), (), activation_kind(se.gate), f"{prefix}.se")
