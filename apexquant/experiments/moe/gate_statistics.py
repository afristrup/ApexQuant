"""Per-expert routing statistics of a model on a set of windows."""

from __future__ import annotations

import torch

from .experts_modules import experts_modules


@torch.no_grad()
def gate_statistics(model, windows: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
    """``usage`` (routed token share) and ``energy`` (mean ``g^2 ||x||^2``) per expert."""
    stats, handles = {}, []
    for name, mod in experts_modules(model).items():
        count = mod.gate_up_proj.shape[0]
        stats[name] = {"usage": torch.zeros(count + 1), "energy": torch.zeros(count + 1)}

        def hook(module, args, name=name):
            hidden, idx, weights = args[0], args[1], args[2]
            norm2 = hidden.float().square().sum(-1, keepdim=True).expand_as(weights)
            flat = idx.reshape(-1).cpu()
            stats[name]["usage"].index_add_(0, flat, torch.ones(flat.numel()))
            stats[name]["energy"].index_add_(0, flat, (weights.float().square() * norm2).reshape(-1).cpu())

        handles.append(mod.register_forward_pre_hook(hook))
    try:
        device = next(model.parameters()).device
        for window in windows:
            model(window.unsqueeze(0).to(device))
    finally:
        for h in handles:
            h.remove()
    tokens = windows.numel()
    return {n: {k: v[:-1] / tokens for k, v in s.items()} for n, s in stats.items()}
