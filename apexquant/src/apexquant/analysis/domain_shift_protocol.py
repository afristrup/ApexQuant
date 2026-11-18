"""Source versus protected-target quantization evaluation protocol."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn

from .distribution_shift import distribution_shift


def measure_domain_shift(
    reference: nn.Module,
    candidate: nn.Module,
    batches: Iterable[torch.Tensor],
    *,
    forward: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
    logits: bool = True,
) -> dict[str, float]:
    """Measure candidate drift on fixed, held-out input batches.

    The batches are measurement-only. They must not be used to choose scales,
    codebooks, pass counts, or calibration parameters. Batches are concatenated
    before metrics are computed, so each token contributes equally.
    """
    def default_forward(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        output = model(batch)
        return getattr(output, "logits", output)

    run = forward or default_forward
    ref_parts, cand_parts = [], []
    with torch.no_grad():
        for batch in batches:
            ref_parts.append(run(reference, batch).detach().float().cpu())
            cand_parts.append(run(candidate, batch).detach().float().cpu())
    if not ref_parts:
        raise ValueError("batches must contain at least one evaluation batch")
    return distribution_shift(torch.cat(ref_parts), torch.cat(cand_parts), logits=logits)
