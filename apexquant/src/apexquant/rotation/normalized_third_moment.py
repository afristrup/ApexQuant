"""Spikiness statistic that decides when one Hadamard rotation is not enough."""

from __future__ import annotations

import torch


def normalized_third_moment(x: torch.Tensor) -> torch.Tensor:
    """Return ``sqrt(d) * sum_i |v_i|^3`` for ``v = x / ||x||`` over the last axis.

    The scalar Hadamard distortion bound grows with ``sum_i |v_i|^3``. Scaling by
    ``sqrt(d)`` makes the statistic dimension-free: about 1.6 for a Gaussian-like
    direction, ``sqrt(d / k)`` for a flat ``k``-sparse one, ``sqrt(d)`` for an axis.
    A zero vector returns 0.
    """
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (x / norm).abs().pow(3).sum(dim=-1) * x.shape[-1] ** 0.5
