"""Paired bootstrap of the fine-tune's clinical gain over the base model."""

from __future__ import annotations

import torch

from .task_items import task_items


def significance_gate(ft_dir: str, base_dir: str, tasks: tuple[str, ...],
                      draws: int = 10000, seed: int = 0) -> dict:
    """Mean over tasks of per-task accuracy gain, with a 95% paired bootstrap interval.

    Items are resampled within each task and paired by ``doc_id``. The gate
    passes only if the whole interval lies above zero.
    """
    g = torch.Generator().manual_seed(seed)
    gains = torch.zeros(draws)
    point = 0.0
    for task in tasks:
        ft, base = (task_items(d, task) for d in (ft_dir, base_dir))
        if ft.keys() != base.keys():
            raise RuntimeError(f"{task}: fine-tune and base scored different items")
        diff = torch.tensor([ft[k] - base[k] for k in sorted(ft)])
        point += float(diff.mean()) / len(tasks)
        idx = torch.randint(len(diff), (draws, len(diff)), generator=g)
        gains += diff[idx].mean(1) / len(tasks)
    lo, hi = (float(q) for q in gains.quantile(torch.tensor([0.025, 0.975])))
    return {"gain": point, "ci95": [lo, hi], "passes": lo > 0}
