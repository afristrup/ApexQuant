"""The moment description of a tensor arriving at a convolution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InputSource:
    """Where a convolution's input moments come from, as the graph walk carries it.

    Attributes:
        case: Input-moment case letter of section 4.4.
        producers: Dotted names of the BatchNorms summed into the pre-activation.
        identity_from: Dotted names of nodes whose output moments are reused.
        activation: Activation-kind name applied to the sum.
        node: Dotted name of the module producing this tensor.
    """

    case: str
    producers: tuple[str, ...]
    identity_from: tuple[str, ...]
    activation: str
    node: str
