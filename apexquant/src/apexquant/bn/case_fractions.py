"""Report how a model's convolutions split across the four input-moment cases."""

from __future__ import annotations

from collections.abc import Sequence

from .case_letters import CASE_LETTERS
from .conv_bn_site import ConvBnSite

PERCENT = 100.0


def case_fractions(sites: Sequence[ConvBnSite]) -> dict[str, float]:
    """Return the percentage of ``sites`` in each case letter.

    Section 8.7 asks for the case-D count to be published next to the accuracy,
    because squeeze-excitation gates decide it and the reader needs to see how
    much of the model QBN declined to touch.

    Args:
        sites: The records from :func:`conv_bn_sites`.

    Returns:
        Case letter to percentage of the sites, one entry per letter, summing to
        100 for a non-empty input.

    Raises:
        ValueError: ``sites`` is empty, which means the adapter resolved nothing.
    """
    if not sites:
        raise ValueError("no sites to report on")
    return {letter: PERCENT * sum(s.case == letter for s in sites) / len(sites)
            for letter in CASE_LETTERS}
