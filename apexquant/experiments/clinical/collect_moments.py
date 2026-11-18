"""Input second moments and an activation sample for every distinct linear input."""

from __future__ import annotations

import os

import torch

from .input_key import input_key
from .target_linears import target_linears


@torch.no_grad()
def collect_moments(model, windows: torch.Tensor, out_dir: str, sample_tokens: int,
                    seed: int = 0, only: set[str] | None = None) -> None:
    """Write ``{key}.pt`` holding ``sigma = E[x x^T]`` and ``sample`` rows of ``x``.

    The sample takes the same random positions from every window, so all keys
    see the same tokens and per-sample norms can be compared across layers.
    """
    os.makedirs(out_dir, exist_ok=True)
    keys = sorted({input_key(n) for n in target_linears(model)})
    if only is not None:
        keys = [k for k in keys if k in only]
        if len(keys) != len(only):
            raise KeyError(f"unknown moment keys {sorted(only - set(keys))}")
    modules = dict(model.named_modules())
    per_window = max(1, sample_tokens // windows.shape[0])
    picks = torch.randperm(windows.shape[1], generator=torch.Generator().manual_seed(seed))[:per_window]
    sigma, sample, count = {}, {k: [] for k in keys}, [0]

    def hook(key):
        def run(_, args):
            x = args[0].reshape(-1, args[0].shape[-1]).float()
            if key not in sigma:
                sigma[key] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
            sigma[key].addmm_(x.T, x)
            sample[key].append(x[picks.to(x.device)].half().cpu())
        return run

    handles = [modules[k].register_forward_pre_hook(hook(k)) for k in keys]
    try:
        device = next(model.parameters()).device
        for window in windows:
            model(window.unsqueeze(0).to(device))
            count[0] += window.numel()
    finally:
        for h in handles:
            h.remove()
    for key in keys:
        torch.save({"sigma": (sigma.pop(key) / count[0]).cpu(),
                    "sample": torch.cat(sample.pop(key))},
                   os.path.join(out_dir, f"{key}.pt"))
