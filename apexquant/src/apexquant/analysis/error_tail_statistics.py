"""Measure whether the reconstruction error is heavy-tailed, as HQQ assumes."""

from __future__ import annotations

import math

import torch


def error_tail_statistics(error: torch.Tensor, top_fraction: float = 0.001,
                          shape_grid: int = 241) -> dict[str, float]:
    """Summarize the tail of a per-weight reconstruction error in three numbers.

    Our positioning against HQQ is a distributional claim: they handle outliers
    in the *loss*, with an ``l_p`` norm justified by a hyper-Laplacian model of
    quantization error, and we handle them in the *geometry*, since after the
    rotation and normalization the source is supported on ``[-1, 1]`` and
    log-concave, so ``delta w`` is bounded coordinatewise by the widest cell of a
    codebook matched to a compactly supported density. That is an argument, not a
    measurement, and this function is the measurement. Run it on
    ``weight - dequantized`` in the *original* frame, at each pass.

    Read it against a Gaussian, which is the neutral reference: excess kurtosis
    ``0``, shape ``2``, and a top-``0.1%`` share of about ``1.7%``. Heavier than
    Gaussian pushes kurtosis and the top share up and the shape parameter down,
    and a hyper-Laplacian in HQQ's sense would land near ``shape < 1``. Lighter
    than Gaussian, which is what a compactly supported source predicts, pushes
    the shape above ``2``.

    Args:
        error: Any shape; flattened. Typically ``weight - dequantize_weight(qw)``.
        top_fraction: Tail mass fraction for the concentration statistic. The
            default reports the share of total squared error carried by the
            largest ``0.1%`` of coordinates.
        shape_grid: Candidate exponents for the moment-matched generalized
            Gaussian fit, swept over ``[0.2, 12]``.

    Returns:
        Dict with ``excess_kurtosis``, ``top_share``, ``gg_shape`` and ``count``.
        ``gg_shape`` is fitted by matching the kurtosis of a generalized
        Gaussian, which is monotone in the exponent and so inverts uniquely; it
        is a summary of the same tail the other two numbers describe, not an
        independent estimate.
    """
    e = error.detach().to(torch.float64).flatten()
    n = e.numel()
    if n < 2:
        raise ValueError("error must contain at least two elements")

    centered = e - e.mean()
    var = centered.pow(2).mean()
    if var <= 0:
        return {"excess_kurtosis": 0.0, "top_share": 0.0, "gg_shape": math.inf,
                "count": float(n)}

    kurtosis = (centered.pow(4).mean() / var.pow(2)).item()
    excess = kurtosis - 3.0

    # Share of squared error carried by the largest coordinates. This is the
    # statistic that actually matters for the claim: a heavy tail means a few
    # coordinates own the damage.
    k = max(1, int(round(top_fraction * n)))
    sq = centered.pow(2)
    top_share = (sq.topk(k).values.sum() / sq.sum()).item()

    # Generalized Gaussian kurtosis is G(5/p)G(1/p)/G(3/p)^2, strictly decreasing
    # in p, so a grid sweep with a monotone lookup is exact enough and needs no
    # special-function inverse.
    p = torch.linspace(0.2, 12.0, shape_grid, dtype=torch.float64)
    lg = torch.lgamma
    kurt_p = torch.exp(lg(5.0 / p) + lg(1.0 / p) - 2.0 * lg(3.0 / p))
    gg_shape = p[(kurt_p - kurtosis).abs().argmin()].item()

    return {
        "excess_kurtosis": excess,
        "top_share": top_share,
        "gg_shape": gg_shape,
        "count": float(n),
    }
