"""Mean and standard deviation of the pre-activation arriving at one convolution."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .bn_output_moments import bn_output_moments
from .case_letters import CASE_A, CASE_B, CORRECTABLE_CASES
from .conv_bn_site import ConvBnSite

#: Case A is one BatchNorm. Case B is a sum, which needs at least two branches.
CASE_A_BRANCHES = 1
CASE_B_MIN_BRANCHES = 2


def pre_activation_moments(model: nn.Module, site: ConvBnSite,
                           recorded: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(m, sigma)`` of the tensor ``site``'s activation is applied to.

    Case A is one BatchNorm read through :func:`bn_output_moments`, so
    ``m = beta`` and ``sigma = |gamma| sqrt(v/(v+eps))``.

    Case B is a residual sum. Section 4.4 adds branch means and branch variances,

        m = sum_r m_r,   sigma^2 = sum_r sigma_r^2

    which is DFQ section 5.1.2's rule and Brock et al.'s variance recursion, and
    whose zero-cross-covariance assumption is not checkable from any BatchNorm
    buffer. Section 8.2 owns that risk; this function only implements the rule.
    A branch that ends in a BatchNorm contributes that BatchNorm's output
    moments. A branch separated from its BatchNorm by a nonlinearity cannot be
    reduced to a ``(beta, gamma, v)`` triple, so it contributes the *recorded*
    output moments of the node it copies, converted from the raw second moment to
    a variance: ``m += e_n`` and ``sigma^2 += s_n - e_n^2``.

    The activation is deliberately not applied here. Section 4.4 sums first and
    applies the activation formula to the sum, and splitting the two steps is
    what keeps that order impossible to get wrong.

    Args:
        model: The model the site's dotted names address.
        site: One record from :func:`conv_bn_sites`, in a correctable case.
        recorded: Node name to the ``(e, s)`` output moments already computed for
            that node. Sites are processed in forward order, so every name in
            ``site.identity_from`` is present by the time it is needed.

    Returns:
        Tuple ``(m, sigma)`` of shape ``(C_in,)``. Every addend is a variance and
        so non-negative, and the sum is floored at zero before the square root
        only to absorb the last-bit cancellation an identity branch's
        ``s_n - e_n^2`` can produce.

    Raises:
        ValueError: The case is not correctable, the producer count contradicts
            the case letter, an identity branch has no recorded moments, or two
            branches disagree about the channel count.
    """
    if site.case not in CORRECTABLE_CASES:
        raise ValueError(f"{site.conv} is case {site.case}; only "
                         f"{CORRECTABLE_CASES} carry an input model")
    branches = len(site.producers) + len(site.identity_from)
    if site.case == CASE_A and (len(site.producers) != CASE_A_BRANCHES
                                or branches != CASE_A_BRANCHES):
        raise ValueError(f"case {CASE_A} site {site.conv} must have exactly "
                         f"{CASE_A_BRANCHES} producer, got {site.producers} "
                         f"and {site.identity_from}")
    if site.case == CASE_B and branches < CASE_B_MIN_BRANCHES:
        raise ValueError(f"case {CASE_B} site {site.conv} is a sum of "
                         f"{branches} branches, which is not a sum")

    mean: torch.Tensor | None = None
    variance: torch.Tensor | None = None
    for name in site.producers:
        m, sigma = bn_output_moments(model.get_submodule(name))
        mean = m if mean is None else mean + m
        variance = sigma * sigma if variance is None else variance + sigma * sigma
    for name in site.identity_from:
        if name not in recorded:
            raise ValueError(f"{site.conv} copies the moments of node {name!r}, "
                             "which has not been recorded; process sites in "
                             "forward order")
        e, s = recorded[name]
        mean = e if mean is None else mean + e
        branch = s - e * e
        variance = branch if variance is None else variance + branch
    if mean is None or variance is None or mean.shape != variance.shape:
        raise ValueError(f"{site.conv} has no branch to read moments from")
    return mean, variance.clamp_min(0.0).sqrt()
