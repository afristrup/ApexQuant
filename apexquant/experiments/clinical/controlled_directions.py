"""Unit-norm rank-r updates along directions clinical text excites and generic text does not."""

from __future__ import annotations

import os

import torch

from .generalized_top import generalized_top
from .input_key import input_key


def controlled_directions(model, source_dir: str, target_dir: str, layers, module: str,
                          rank: int, seed: int) -> dict[str, torch.Tensor]:
    """``B A`` per chosen layer, scaled so that ``||B A||_F = ||W||_F``.

    Rows of ``A`` are the top ``rank`` generalized eigenvectors of
    ``(Sigma_t, Sigma_s)``; ``B`` is Gaussian with a fixed seed.
    """
    g = torch.Generator().manual_seed(seed)
    modules = dict(model.named_modules())
    out = {}
    for layer in layers:
        name = f"model.layers.{layer}.{module}"
        key = input_key(name)
        s_s = torch.load(os.path.join(source_dir, f"{key}.pt"))["sigma"].cuda()
        s_t = torch.load(os.path.join(target_dir, f"{key}.pt"))["sigma"].cuda()
        _, a = generalized_top(s_t, s_s, rank)
        w = modules[name].weight
        b = torch.randn(w.shape[0], rank, generator=g).cuda()
        update = b @ a
        out[f"{name}.weight"] = (update * (w.float().norm() / update.norm())).cpu()
    return out
