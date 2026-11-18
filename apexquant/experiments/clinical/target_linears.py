"""The decoder linear layers every arm quantizes."""

from __future__ import annotations

import torch


def target_linears(model) -> dict[str, torch.nn.Linear]:
    """Every ``nn.Linear`` inside a decoder layer, keyed by module name."""
    found = {name: mod for name, mod in model.named_modules()
             if isinstance(mod, torch.nn.Linear) and ".layers." in name}
    if not found:
        raise RuntimeError("no decoder linear layers found")
    return found
