"""Sinkhorn-Knopp row/column scales, applied *before* the rotation (SINQ)."""

from __future__ import annotations

import torch


def sinkhorn_scales(weight: torch.Tensor, iters: int = 10,
                    eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(row, col)`` scales that equalize per-row and per-column variance.

    SINQ (Mueller et al., arXiv:2509.22944) observes that a *single* scale per
    group forces outliers and their row-mates to share a quantization grid, and
    that a second scale along the other axis fixes it. It finds both with a
    Sinkhorn-Knopp iteration that drives the matrix towards equal row and column
    variances ("matrix imbalance"), and reports that the recovered column scales
    are ``predictive of the typical activation magnitudes the matrix received
    during training`` -- an estimate of ``diag(H)`` extracted from the weights
    alone, with no calibration data.

    That is exactly the structure our rotation destroys, so the composition only
    works in one order. The randomized Hadamard *mixes columns*, so a per-column
    scale applied after it is meaningless; the normalization has to come first.
    Applied first it is harmless to the theory: ``row`` and ``col`` are
    deterministic functions of ``weight``, hence independent of the fresh
    rotations, so re-isotropization still applies at every pass and the recursion
    is still an L2 recursion in the normalized coordinates. What changes is the
    proxy loss, which becomes weighted by the column diagonal.

    The factorization is ``weight[o, i] ~= row[o] * normalized[o, i] * col[i]``,
    so the deployed form costs one elementwise activation scale and one output
    scale, and the input-side diagonal folds into the preceding layer the way
    SmoothQuant's does.

    Args:
        weight: Dense weight ``(out_features, in_features)``.
        iters: Sinkhorn-Knopp sweeps. SINQ reports zero to negligible overhead;
            10 is well past convergence for transformer weight shapes.
        eps: Floor on every scale, so an all-zero row or column is a no-op
            rather than a division by zero.

    Returns:
        ``(row, col)`` float32 tensors of shape ``(out_features,)`` and
        ``(in_features,)``. Dividing ``weight`` by their outer product gives a
        matrix with near-unit row and column standard deviation.
    """
    if weight.ndim != 2:
        raise ValueError("weight must be 2-D (out_features, in_features)")
    if iters < 0:
        raise ValueError("iters must be >= 0")

    w = weight.to(torch.float32)
    row = torch.ones(w.shape[0], dtype=torch.float32, device=w.device)
    col = torch.ones(w.shape[1], dtype=torch.float32, device=w.device)

    for _ in range(iters):
        # Row sweep: rescale so every row of the *currently normalized* matrix has
        # unit std. Both scales have to divide out here; dropping either one turns
        # the update into a compounding multiply that diverges within a few sweeps.
        r = (w / (row[:, None] * col)).pow(2).mean(dim=1).sqrt().clamp_min(eps)
        row = row * r
        # Column sweep: same along the other axis, under the updated rows.
        c = (w / (row[:, None] * col)).pow(2).mean(dim=0).sqrt().clamp_min(eps)
        col = col * c

    # Fix the gauge: the factorization is invariant under row *= t, col /= t, so
    # pin the columns to geometric mean 1 and let the rows carry the overall
    # magnitude. Without this the split between the two vectors is arbitrary and
    # the stored scales drift over repeated quantize/dequantize round trips.
    gauge = col.clamp_min(eps).log().mean().exp()
    return row * gauge, col / gauge
