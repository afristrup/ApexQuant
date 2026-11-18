"""Every (model, policy, average bits) row of E6."""

from __future__ import annotations

from .settings import AVERAGE_BITS, MODELS, PMAX, POLICIES


def policy_grid() -> list[tuple[str, str, int]]:
    """Uniform depth at every integer pass count; the other policies at 3, 4 and 5 bits."""
    rows = []
    for model in MODELS:
        rows += [(model, "uniform", 2 * p) for p in range(1, PMAX + 1)]
        rows += [(model, policy, bits) for policy in POLICIES if policy != "uniform"
                 for bits in AVERAGE_BITS]
    return rows
