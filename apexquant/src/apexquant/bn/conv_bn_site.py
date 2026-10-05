"""One convolution's BatchNorm neighbourhood, as QBN's correction needs it."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConvBnSite:
    """What the QBN driver must know about a single convolution.

    Attributes:
        conv: Dotted module name of the convolution.
        bn: Dotted name of the BatchNorm that immediately follows it, or None
            when the convolution's output is not normalized.
        case: Input-moment case letter, ``CASE_A`` to ``CASE_D`` of section 4.4.
        producers: Dotted names of the BatchNorms whose outputs are summed to
            form this convolution's pre-activation input. One name in case A,
            one or more in case B, empty in cases C and D.
        identity_from: Dotted names of the nodes whose recorded output moments
            an identity branch of a case-B sum copies, for branches that a
            nonlinearity separates from their BatchNorm. Empty otherwise.
        activation: Activation-kind name applied between the producers and this
            convolution.
        source: Dotted name of the module producing the input tensor. Diagnostic
            only -- it is what lets a case-D assignment be explained.
    """

    conv: str
    bn: str | None
    case: str
    producers: tuple[str, ...]
    identity_from: tuple[str, ...]
    activation: str
    source: str
