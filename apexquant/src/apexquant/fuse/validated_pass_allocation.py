"""Accept a per-layer allocation only if it actually beats uniform."""

from __future__ import annotations

from typing import Callable, Mapping


def validated_pass_allocation(
    candidate: Mapping[str, int],
    fallback: Mapping[str, int],
    score: Callable[[Mapping[str, int]], float],
    *,
    margin: float = 1.0,
) -> dict[str, int]:
    """Return ``candidate`` if it scores better than ``fallback``, else ``fallback``.

    Every ingredient of a predicted allocation can be right while the allocation
    is still wrong, and on real models it happens. Three separate measurements
    say why. The relative error is not the culprit: the decay law is accurate
    enough that ``t_l = rho**P`` is the same for every layer to within a percent,
    so it carries no ranking information and cannot be mis-ranked. Neither is the
    coefficient estimate: replacing the noise-fitted coefficient with the damage
    measured by really quantizing each layer changes the ranking and not the
    outcome. What remains is the summation itself. Per-layer damages are measured
    one layer at a time and then added, and at two bits with every layer
    quantized at once that additivity is not accurate enough to justify moving
    budget between layers whose true sensitivities are close together.

    So the allocation is a hypothesis, and this function tests it. Score the
    proposed allocation and the uniform fallback end to end and keep the winner.
    It costs two evaluations rather than two hundred, it inherits data-freeness
    from ``score``, and it converts the allocator from something that helps on
    some models and hurts on others into something that is never worse than
    uniform and better where the structure is real.

    The check has to be run on inputs the allocation was not chosen against, or
    it certifies its own overfitting. With a random-token probe that means a
    different seed; the caller owns that choice because the caller owns
    ``score``.

    Args:
        candidate: ``{layer: passes}`` from :func:`knapsack_pass_allocation`.
        fallback: ``{layer: passes}`` to fall back to, normally uniform at the
            same budget. Must cover the same layers as ``candidate``, since
            otherwise the two are not being compared at equal cost.
        score: Maps a pass map to a divergence, lower being better. Called
            exactly twice.
        margin: Required improvement factor. ``1.0`` keeps the candidate on any
            improvement; ``1.05`` demands five percent and is the setting to use
            when ``score`` is itself noisy, since it stops the fallback being
            abandoned for a difference the probe cannot resolve.

    Returns:
        Either ``candidate`` or ``fallback``, as a plain dict.
    """
    if set(candidate) != set(fallback):
        raise ValueError("candidate and fallback must cover the same layers")
    if margin < 1.0:
        raise ValueError("margin must be >= 1")

    candidate_score = float(score(candidate))
    fallback_score = float(score(fallback))
    if candidate_score * margin < fallback_score:
        return dict(candidate)
    return dict(fallback)
