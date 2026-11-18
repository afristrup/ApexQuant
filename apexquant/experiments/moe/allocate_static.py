"""Exact static pass allocation over the experts of one layer."""

from __future__ import annotations

import torch


def allocate_static(weight: torch.Tensor, u2: torch.Tensor, budget: int) -> torch.Tensor:
    """Passes ``p_e in 1..P`` minimizing ``sum_e weight_e U2_e(p_e)`` with ``sum p_e <= budget``.

    A multiple-choice knapsack solved by dynamic programming over the extra
    passes beyond one each; ``u2[e, p - 1]`` is the error after ``p`` passes.
    """
    experts, pmax = u2.shape
    extra = budget - experts
    if not 0 <= extra <= experts * (pmax - 1):
        raise ValueError(f"budget {budget} outside [{experts}, {experts * pmax}]")
    cost = (weight[:, None] * u2).double().cpu()
    inf = float("inf")
    best = torch.full((experts + 1, extra + 1), inf, dtype=torch.float64)
    best[0, 0] = 0.0
    choice = torch.zeros(experts, extra + 1, dtype=torch.long)
    for e in range(experts):
        for used in range(extra + 1):
            for add in range(min(pmax - 1, used) + 1):
                value = best[e, used - add] + cost[e, add]
                if value < best[e + 1, used]:
                    best[e + 1, used] = value
                    choice[e, used] = add
    used = int(best[experts].argmin())
    passes = torch.ones(experts, dtype=torch.long)
    for e in range(experts - 1, -1, -1):
        passes[e] += choice[e, used]
        used -= int(choice[e, used])
    return passes
