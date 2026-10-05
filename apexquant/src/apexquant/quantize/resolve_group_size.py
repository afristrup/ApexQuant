"""Resolve the group size actually used for a weight of a given width."""

from __future__ import annotations


def resolve_group_size(in_features: int, group_size: int) -> int:
    """Return the group size that will actually be used for ``in_features`` cols.

    A layer wider than ``group_size`` uses it unchanged. A *narrower* layer
    shrinks the group to the largest power of two ``<= in_features`` (minimum 2)
    so the fast Walsh-Hadamard rotation still applies and we never pad a small
    row up to a full group (which would inflate its stored bits). This is the one
    place that decision lives; both :func:`turboquantize_weight` and the fuser's
    eligibility check call it, so they can never disagree about which layers are
    fusable.

    Args:
        in_features: Input dimension of the weight.
        group_size: Requested group size (a power of two ``>= 2``).

    Returns:
        The effective group size (a power of two ``>= 2``).
    """
    if in_features >= group_size:
        return group_size
    gs = 1
    while gs * 2 <= in_features:
        gs *= 2
    return max(gs, 2)
