"""Combine the branches of a residual sum into one input description."""

from __future__ import annotations

from .activation_kinds import ACT_IDENTITY
from .case_letters import CASE_B, CASE_C, CASE_D
from .input_source import InputSource


def residual_input_source(branch: InputSource, shortcut: InputSource,
                          activation: str, node: str) -> InputSource:
    """Describe ``activation(branch + shortcut)`` as a case-B input, or fail to case D.

    Section 4.4's case-B rule adds branch means and branch variances, so a branch
    may be merged into the producer list only when nothing nonlinear stands
    between its BatchNorm and the sum. A branch that a nonlinearity does separate
    is recorded in ``identity_from`` instead: its moments must be read off the
    node it copies, which is exactly DFQ's "reuse the recorded moments" rule. A
    branch that is itself uncorrectable poisons the sum.

    Args:
        branch: Input description of the residual branch.
        shortcut: Input description of the skip branch.
        activation: Activation-kind name applied after the sum.
        node: Dotted name of the module whose output this sum is.

    Returns:
        A case-B description, or a case-D one when a branch is uncorrectable.
    """
    producers: tuple[str, ...] = ()
    identity_from: tuple[str, ...] = ()
    for part in (branch, shortcut):
        if part.case in (CASE_C, CASE_D):
            return InputSource(CASE_D, (), (), activation, node)
        if part.activation == ACT_IDENTITY:
            producers += part.producers
            identity_from += part.identity_from
        else:
            identity_from += (part.node,)
    return InputSource(CASE_B, producers, identity_from, activation, node)
