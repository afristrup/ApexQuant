"""Size an additive update so it moves the model by a fixed KL on held-out windows."""

from __future__ import annotations

import math

import torch

from .token_kl import token_kl


@torch.no_grad()
def scale_to_kl(reference, candidate, updates: dict[str, torch.Tensor],
                windows: torch.Tensor, target: float, steps: int = 14) -> float:
    """The factor ``t`` with ``KL(ref || ref + t U) = target``, by bisection in ``log t``.

    ``candidate`` starts equal to ``reference`` and ends holding ``ref + t U``.
    """
    params = dict(candidate.named_parameters())
    pristine = {n: params[n].data.clone() for n in updates}

    def apply(t):
        for n, u in updates.items():
            params[n].data = (pristine[n].float() + t * u.to(pristine[n].device)).to(pristine[n].dtype)
        return token_kl(reference, candidate, windows)["kl_mean"]

    lo, hi = math.log(1e-4), math.log(1.0)
    if apply(math.exp(hi)) < target:
        raise RuntimeError(f"even t=1 moves KL by less than {target}")
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if apply(math.exp(mid)) < target else (lo, mid)
    t = math.exp(hi)
    apply(t)
    return t
