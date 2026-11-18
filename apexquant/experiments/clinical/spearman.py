"""Spearman rank correlation without a scipy dependency."""

from __future__ import annotations

import torch


def spearman(x: list[float], y: list[float]) -> float:
    """Pearson correlation of the ranks of ``x`` and ``y``, ties broken by order."""
    if len(x) != len(y) or len(x) < 3:
        raise ValueError(f"need two equal lists of at least 3 values, got {len(x)}, {len(y)}")
    rx = torch.tensor(x).argsort().argsort().double()
    ry = torch.tensor(y).argsort().argsort().double()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()))
