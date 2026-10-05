"""Hand-written topology adapter for timm's ResNet family."""

from __future__ import annotations

from torch import nn

from .activation_kind import activation_kind
from .activation_kinds import ACT_IDENTITY, ACT_IMAGE
from .case_letters import CASE_A, CASE_C, CASE_D
from .conv_bn_site import ConvBnSite
from .input_source import InputSource
from .residual_input_source import residual_input_source
from .site_from_source import site_from_source

STAGE_NAMES = ("layer1", "layer2", "layer3", "layer4")
IMAGE_SOURCE = "input"
DOWNSAMPLE_LEN = 2
_PASSTHROUGH_ATTRS = ("drop_block", "aa", "se", "drop_path")


def resnet_conv_bn_sites(model: nn.Module) -> list[ConvBnSite]:
    """Resolve every convolution of a ResNet to its BatchNorm neighbourhood.

    The bottleneck is ``conv1-bn1-relu-conv2-bn2-relu-conv3-bn3-add-relu``, so
    each block's second and third convolutions read a single BatchNorm through a
    ReLU (case A) while its first convolution and the downsample convolution read
    the previous block's post-addition output (case B). The stem convolution
    reads the image (case C) and the two convolutions that open ``layer1`` read
    the max pool, which section 4.4 puts in case D.

    Args:
        model: A timm ``ResNet`` in eval mode. Weights are never read.

    Returns:
        One :class:`ConvBnSite` per ``nn.Conv2d``, in forward order.
    """
    sites = [ConvBnSite("conv1", "bn1", CASE_C, (), (), ACT_IMAGE, IMAGE_SOURCE)]
    source = InputSource(CASE_A, ("bn1",), (), activation_kind(model.act1), "act1")
    if not isinstance(model.maxpool, nn.Identity):
        source = InputSource(CASE_D, (), (), source.activation, "maxpool")
    for stage_name in STAGE_NAMES:
        for index, block in enumerate(getattr(model, stage_name)):
            source = _block_sites(sites, f"{stage_name}.{index}", block, source)
    return sites


def _block_sites(sites: list[ConvBnSite], prefix: str, block: nn.Module,
                 source: InputSource) -> InputSource:
    """Append one residual block's sites and return its output description."""
    for attr in _PASSTHROUGH_ATTRS:
        module = getattr(block, attr, None)
        if module is not None and not isinstance(module, nn.Identity):
            raise TypeError(f"{prefix}.{attr} is {type(module).__name__}, not a "
                            f"pass-through; this adapter does not model it")
    shortcut, depth = source, 0
    while hasattr(block, f"conv{depth + 1}"):
        depth += 1
    for position in range(1, depth + 1):
        sites.append(site_from_source(f"{prefix}.conv{position}",
                                      f"{prefix}.bn{position}", source))
        tail = ACT_IDENTITY if position == depth else \
            activation_kind(getattr(block, f"act{position}"))
        source = InputSource(CASE_A, (f"{prefix}.bn{position}",), (), tail,
                             f"{prefix}.bn{position}")
    if block.downsample is not None:
        shortcut = _downsample_sites(sites, prefix, block.downsample, shortcut)
    return residual_input_source(source, shortcut,
                                 activation_kind(getattr(block, f"act{depth}")),
                                 prefix)


def _downsample_sites(sites: list[ConvBnSite], prefix: str, downsample: nn.Module,
                      source: InputSource) -> InputSource:
    """Append the projection shortcut's site and return its output description."""
    children = list(downsample)
    if len(children) != DOWNSAMPLE_LEN or not isinstance(children[0], nn.Conv2d) \
            or not isinstance(children[1], nn.BatchNorm2d):
        raise TypeError(f"{prefix}.downsample is not a Conv2d-BatchNorm2d pair")
    bn = f"{prefix}.downsample.1"
    sites.append(site_from_source(f"{prefix}.downsample.0", bn, source))
    return InputSource(CASE_A, (bn,), (), ACT_IDENTITY, bn)
