"""Encode every expert once to its deepest prefix and record its error per prefix."""

from __future__ import annotations

import torch

from .apex_prefixes import apex_prefixes
from .settings import GROUP, PMAX


@torch.no_grad()
def encode_experts(experts, stage, seed: int) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Prefix stacks ``gate_up[p]``, ``down[p]`` of shape ``(E, out, in)`` and ``U2[e, p]``.

    ``U2[e, p]`` is ``||W_e - W_e^(p+1)||_F^2`` summed over the expert's two
    matrices, the stored error record of a ``p + 1``-pass read. A matrix whose
    fan-in does not divide the group is grouped along its fan-out instead, as in
    ``clinical.grouped_axis``, so every pass costs exactly ``2 + 16 / 512`` bits.
    """
    device = experts.gate_up_proj.device
    gate_up = [torch.empty_like(experts.gate_up_proj) for _ in range(PMAX)]
    down = [torch.empty_like(experts.down_proj) for _ in range(PMAX)]
    u2 = torch.zeros(experts.gate_up_proj.shape[0], PMAX, device=device)
    for stacks, tensor in ((gate_up, experts.gate_up_proj), (down, experts.down_proj)):
        for e, w in enumerate(tensor.data):
            w = w.cuda()
            if w.shape[1] % GROUP == 0:
                prefixes = apex_prefixes(w, stage, PMAX, GROUP, seed)
            elif w.shape[0] % GROUP == 0:
                prefixes = [r.T for r in apex_prefixes(w.T.contiguous(), stage, PMAX, GROUP, seed)]
            else:
                raise ValueError(f"neither side of a {tuple(w.shape)} expert divides group {GROUP}")
            for p, rec in enumerate(prefixes):
                stacks[p][e] = rec.to(device)
                u2[e, p] += (rec.float() - w.cuda().float()).square().sum().to(device)
    return gate_up, down, u2
