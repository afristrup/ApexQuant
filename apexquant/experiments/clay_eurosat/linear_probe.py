"""A multinomial logistic probe fit by full-batch L-BFGS."""

from __future__ import annotations

import torch


def linear_probe(x: torch.Tensor, y: torch.Tensor, classes: int,
                 weight_decay: float = 1e-4) -> torch.nn.Linear:
    """Fit on standardized features; the returned layer folds the standardization in."""
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    z = (x - mean) / std
    head = torch.nn.Linear(x.shape[1], classes)
    opt = torch.optim.LBFGS(head.parameters(), max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(z), y) + weight_decay * head.weight.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    folded = torch.nn.Linear(x.shape[1], classes)
    with torch.no_grad():
        folded.weight.copy_(head.weight / std)
        folded.bias.copy_(head.bias - (head.weight * mean / std).sum(1))
    return folded
