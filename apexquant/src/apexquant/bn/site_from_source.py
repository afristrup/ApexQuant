"""Turn a carried input description into the record for one convolution."""

from __future__ import annotations

from .conv_bn_site import ConvBnSite
from .input_source import InputSource


def site_from_source(conv: str, bn: str | None, source: InputSource) -> ConvBnSite:
    """Build the :class:`ConvBnSite` for ``conv`` reading ``source``.

    Args:
        conv: Dotted name of the convolution.
        bn: Dotted name of the BatchNorm that follows it, or None.
        source: The input description the graph walk carries at this point.

    Returns:
        The record, with the source's case, producers and activation copied in.
    """
    return ConvBnSite(conv=conv, bn=bn, case=source.case,
                      producers=source.producers,
                      identity_from=source.identity_from,
                      activation=source.activation, source=source.node)
