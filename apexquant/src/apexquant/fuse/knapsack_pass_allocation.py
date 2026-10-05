"""Optimal per-layer pass counts under a bit budget, data-free."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def knapsack_pass_allocation(
    coefficients: Mapping[str, float],
    sizes: Mapping[str, int],
    rho: float | Mapping[str, float],
    bit_budget: float,
    *,
    bits_per_pass: float,
    max_passes: int = 4,
    min_passes: int = 1,
    resolution: int = 200_000,
    damage: Mapping[str, Mapping[int, float]] | None = None,
) -> dict[str, int]:
    """Choose ``P_l`` per layer to minimize predicted model error under a budget.

    Composing the linearity theorem with the geometric decay law gives a
    predicted end-to-end error that is a closed-form function of the pass counts
    alone,

        ``sum_l alpha_l * rho_l ** (2 * P_l)``,

    while the storage cost is linear in them, ``sum_l size_l * P_l *
    bits_per_pass``. Choosing the pass counts is therefore a resource-allocation
    problem solvable *before any weight is quantized*, and with no calibration
    corpus at all if the coefficients came from a data-free probe. This is the
    data-free replacement for a leave-one-out calibration sweep: HIGGS solve the
    analogous per-layer *bit-width* problem with an integer-programming solver,
    and this is the pass-count version of the same reduction.

    The structure is a multiple-choice knapsack: one choice of ``P_l`` per layer,
    one shared budget. It is tempting to solve it greedily by gain-per-bit, since
    each layer's contribution ``alpha_l * (rho_l**2) ** P`` has diminishing
    returns, and that is exact when every step costs the same. It is *not* exact
    here, because layers have different parameter counts and therefore different
    step costs, so a locally best ratio can leave a budget remainder that no
    later step fits into. We solve it by dynamic programming over a discretized
    budget instead, which is exact on the discretized instance.

    Discretization is the one approximation and it is deliberately one-sided:
    step costs round *up* to the bucket grid and the budget rounds *down*, so the
    returned allocation is always feasible against the true budget, never over
    it. With the default resolution the residual slack is under one part in
    ``resolution`` of the total, far below the granularity of a pass.

    Args:
        coefficients: ``{layer: alpha}`` from :func:`linear_error_coefficients`.
        sizes: ``{layer: parameter count}``. Only layers in *both* mappings are
            allocated.
        rho: Per-pass decay constant, shared or per layer. It depends on the
            stage quantizer and the fan-in, both known in advance.
        bit_budget: Target average bits per weight over the allocated layers.
        bits_per_pass: Stored bits per weight per pass, e.g. ``b + 16/g``.
        max_passes: Upper bound per layer.
        min_passes: Lower bound per layer, granted to every layer before
            anything is optimized, so the floor is a guarantee the budget cannot
            take away.
        resolution: Budget buckets for the DP. Higher is finer and slower;
            runtime is ``O(len(sizes) * max_passes * resolution)`` vectorized.
        damage: Optional ``{layer: {P: divergence}}`` measured directly with the
            real quantizer (:func:`layer_damage_table`). When given it replaces
            the modelled objective ``alpha_l * rho_l**(2P)`` entirely, and
            ``coefficients`` is used only to select layers and ``rho`` is
            ignored. Prefer it: on real weights the decay law is accurate enough
            that ``rho**P`` is the same for every layer to within a percent, so
            the whole ranking rests on the coefficient, and a measured table
            removes both the noise-to-quantizer transfer and the extrapolation
            from the probe levels to the operating point.

    Returns:
        ``{layer: passes}``. Raises if ``min_passes`` alone exceeds the budget,
        since silently returning an over-budget allocation is the one failure
        mode nothing downstream would catch.
    """
    if min_passes < 1 or max_passes < min_passes:
        raise ValueError("need 1 <= min_passes <= max_passes")
    if bits_per_pass <= 0:
        raise ValueError("bits_per_pass must be > 0")
    if resolution < 1:
        raise ValueError("resolution must be >= 1")

    names = [n for n in coefficients if n in sizes]
    if not names:
        raise ValueError("coefficients and sizes share no layer")

    def rho_of(name: str) -> float:
        r = float(rho[name]) if isinstance(rho, Mapping) else float(rho)
        if not 0.0 < r < 1.0:
            raise ValueError(f"rho for {name} must lie in (0, 1)")
        return r

    total_size = sum(sizes[n] for n in names)
    # Work in "weight-passes"; bits_per_pass is a common factor on both sides.
    budget_units = bit_budget * total_size / bits_per_pass
    floor_units = float(total_size * min_passes)
    if floor_units > budget_units * (1 + 1e-9):
        raise ValueError(
            f"min_passes={min_passes} needs "
            f"{floor_units * bits_per_pass / total_size:.3f} bits/weight, "
            f"over the {bit_budget:.3f} budget"
        )

    # Only the passes above the floor are up for allocation.
    span = max_passes - min_passes
    if span == 0:
        return {n: min_passes for n in names}

    free_units = budget_units - floor_units
    if free_units <= 0.0:
        # The floor exactly consumes the budget (the common case when the target
        # is an integer number of uniform passes). There is nothing to allocate,
        # and dividing by a zero-width bucket would produce an unbounded grid.
        return {n: min_passes for n in names}
    # One bucket per resolution step of the *free* budget, so the grid is bounded
    # by construction rather than by the magnitude of the parameter counts.
    unit = free_units / resolution
    budget_buckets = resolution

    # cost[i, k] and value[i, k] for k extra passes on layer i.
    extra = np.arange(span + 1)
    costs = np.empty((len(names), span + 1), dtype=np.int64)
    values = np.empty((len(names), span + 1), dtype=np.float64)
    for i, n in enumerate(names):
        costs[i] = np.ceil(sizes[n] * extra / unit).astype(np.int64)
        if damage is not None:
            row = damage[n]
            missing = [min_passes + k for k in extra if (min_passes + k) not in row]
            if missing:
                raise ValueError(f"damage[{n!r}] lacks pass counts {missing}")
            values[i] = [row[min_passes + k] for k in extra]
        else:
            r2 = rho_of(n) ** 2
            values[i] = coefficients[n] * r2 ** (min_passes + extra)

    inf = np.inf
    dp = np.zeros(budget_buckets + 1, dtype=np.float64)
    dp[1:] = 0.0  # "cost <= k" semantics: spending less is always allowed
    choice = np.zeros((len(names), budget_buckets + 1), dtype=np.int8)

    for i in range(len(names)):
        new = np.full(budget_buckets + 1, inf, dtype=np.float64)
        pick = np.zeros(budget_buckets + 1, dtype=np.int8)
        for k in range(span + 1):
            c = int(costs[i, k])
            if c > budget_buckets:
                continue
            cand = dp[: budget_buckets + 1 - c] + values[i, k]
            better = cand < new[c:]
            new[c:][better] = cand[better]
            pick[c:][better] = k
        dp, choice[i] = new, pick

    # Backtrack from the cheapest optimum.
    k = int(np.argmin(dp))
    passes = {}
    for i in range(len(names) - 1, -1, -1):
        taken = int(choice[i, k])
        passes[names[i]] = min_passes + taken
        k -= int(costs[i, taken])

    return {n: passes[n] for n in names}
