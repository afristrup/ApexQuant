"""Force a level set to be exactly antisymmetric about zero."""

from __future__ import annotations

import torch


def symmetrize_levels(levels: torch.Tensor) -> torch.Tensor:
    """Return ``levels`` made exactly antisymmetric: ``out[i] == -out[n-1-i]``.

    Every density these codebooks are fitted to is symmetric, so the optimal
    level set is antisymmetric. The solvers do not return one. They evaluate the
    density on a finite grid and assign grid points to cells with
    :func:`torch.bucketize`, whose intervals are half-open, so a grid point
    sitting exactly on a cell boundary joins the cell above it and its mirror
    image joins the cell above *its* boundary rather than the one below. The two
    halves then converge to centroids differing by about one grid step, which is
    $5 \\times 10^{-5}$ on the default grid. Closed-form solvers are unaffected:
    :func:`lloyd_max_levels` integrates Gaussian cells analytically from a
    symmetric start and is already exact.

    That residue is far too small to matter to the distortion. It measured
    $2.2 \\times 10^{-8}$ on ``rho**2``, against the four significant figures
    anything reports. It matters because it breaks an *invariance*: the quantizer
    stops commuting with negation, so ``-q(w)`` and ``q(-w)`` differ. The
    residual passes amplify that from harmless to fatal, since a near-boundary
    coordinate can flip to the other codeword and the next pass sees a residual
    an order of magnitude smaller, which turns an absolute perturbation of
    $10^{-5}$ into a relative difference of a few percent by the second pass.

    The concrete casualty is folding a batch normalization into the convolution
    before it. Where the fold constant $\\gamma/\\sqrt{\\sigma^2+\\epsilon}$ is
    positive, folding commutes with quantization exactly, because normalizing each
    group onto the unit sphere divides any positive per-row scalar out. Where
    $\\gamma$ is negative the fold flips the row's sign, and an asymmetric level
    set then quantizes it differently. torchvision's ResNet-50 has 103 negative
    gamma channels of 26,560, and EfficientNet-B0 has 392 of 21,008.

    The average of the level set and its negated mirror is the nearest
    antisymmetric set, is exact in floating point rather than merely close, since
    ``out[n-1-i]`` evaluates the same subtraction negated, and preserves both the
    ascending order and the span.

    Args:
        levels: Sorted 1-D tensor of reconstruction levels.

    Returns:
        A sorted 1-D tensor of the same shape, dtype and device, exactly
        antisymmetric about zero. An odd-length set gets an exact zero in the
        middle, that element being its own mirror.
    """
    if levels.ndim != 1:
        raise ValueError("levels must be 1-D")
    return (levels - levels.flip(0)) / 2
